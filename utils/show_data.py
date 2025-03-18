import math
from typing import Optional

import polyscope
import torch
from torch import Tensor
from torch_geometric import Index
from torch_geometric.data import Data, HeteroData
from torch_geometric.data.storage import NodeStorage, EdgeStorage

# Coloropt colors http://tsitsul.in/blog/coloropt/
COLOROPT_NORMAL = torch.FloatTensor([
    [235, 172, 35],
    [184, 0, 88],
    [0, 140, 249],
    [0, 110, 0],
    [0, 187, 173],
    [209, 99, 230],
    [178, 69, 2],
    [255, 146, 135],
    [89, 84, 214],
    [0, 198, 248],
    [135, 133, 0],
    [0, 167, 108],
    [189, 189, 189],
]) / 255.0
COLOROPT_BRIGHT = torch.FloatTensor([
    [239, 230, 69],
    [233, 53, 161],
    [0, 227, 255],
    [225, 86, 44],
    [83, 126, 255],
    [0, 203, 133],
    [238, 238, 238],
]) / 255.0
COLOROPT_DARK = torch.FloatTensor([
    [0, 89, 0],
    [0, 0, 120],
    [73, 13, 0],
    [138, 3, 79],
    [0, 90, 138],
    [68, 53, 0],
    [88, 88, 88],
]) / 255.0

# Brewer colors
BREWER_SET3 = torch.FloatTensor([
    [141, 211, 199],
    [255, 255, 179],
    [190, 186, 218],
    [251, 128, 114],
    [128, 177, 211],
    [253, 180, 98],
    [179, 222, 105],
    [252, 205, 229],
    [217, 217, 217],
    [188, 128, 189],
    [204, 235, 197],
    [255, 237, 111],
]) / 255.0
BREWER_PAIRED = torch.FloatTensor([
    [66, 206, 227],
    [31, 120, 180],
    [178, 223, 138],
    [51, 160, 44],
    [251, 154, 153],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
    [106, 61, 154],
    [255, 255, 153],
    [177, 89, 40],
]) / 255.0

COLORS = torch.cat([
    COLOROPT_NORMAL,
    BREWER_SET3,
    COLOROPT_BRIGHT,
    COLOROPT_DARK,
    BREWER_PAIRED
    # todo: more colors!
], dim=0)

ID_COLORSCHEME = 'spectral'
SCALAR_COLORSCHEME = 'viridis'


def show_data(data: Data, use_colors: bool = True):
    if data.node_stores[-1].pos.device != 'cpu':
        data = data.clone().to('cpu')

    with torch.no_grad():
        print(data)
        if hasattr(data, 'filename'):
            print(data.filename)
        polyscope.init()
        polyscope.set_up_dir('z_up')
        polyscope.set_front_dir('x_front')
        polyscope.set_ground_plane_mode('shadow_only')

        # Determine the number of batches
        batch_size = 1
        if hasattr(data.node_stores[-1], 'batch'):
            batch_size = torch.max(data.node_stores[-1].batch) + 1

        # Apply offsets to each instance so a whole batch can be viewed at once
        for store in data.node_stores:
            if hasattr(store, 'pos'):
                bbox_size = store.pos.max(dim=0).values - store.pos.min(dim=0).values
                break
        for store in data.node_stores:
            if hasattr(store, 'pos') and hasattr(store, 'batch'):
                y_offset_index = torch.remainder(store.batch, 6)
                x_offset_index = (store.batch - y_offset_index) // 6
                store.pos[:, 0] += x_offset_index * bbox_size[0] * 1.1
                store.pos[:, 1] += y_offset_index * bbox_size[1] * 1.1

        def draw_nodes(nodes: NodeStorage, prefix='', enabled: bool = True, **kwargs):
            # Produce a point cloud with the node's positions
            if not hasattr(nodes, 'pos'):
                return
            pos = nodes.pos
            cloud = polyscope.register_point_cloud(
                prefix,
                pos,
                radius=0.01 / math.sqrt(batch_size),
                enabled=enabled,
                **kwargs
            )

            if hasattr(nodes, 'face'):
                faces = polyscope.register_surface_mesh(
                    prefix + '-face',
                    pos,
                    nodes.face.T.numpy()
                )

            # Broadcast instance-global features to all points
            if hasattr(nodes, 'batch'):
                for key, item in nodes.items():
                    if len(item.shape) == 0:
                        item = item.unsqueeze(0).repeat(batch_size)
                    if isinstance(item, torch.Tensor) and item.shape[0] == batch_size:
                        nodes[key] = item[nodes.batch]

            # Draw each node feature type
            for key, item in nodes.items():
                try:
                    if any(n in key for n in ['edge', 'ptr', 'pos']) or not isinstance(item, torch.Tensor):
                        continue

                    if isinstance(item, Index):
                        mask = torch.zeros([nodes.num_nodes], dtype=item.dtype, device=nodes.pos.device)
                        mask[item] = 1
                        item = mask
                        cloud.add_scalar_quantity(key, item, cmap='reds', enabled=False)
                        continue

                    if item.shape[0] != nodes.num_nodes:
                        if hasattr(nodes, 'batch'):
                            item = item[nodes.batch]
                        else:
                            continue

                    if 'norm' in key:
                        cloud.add_vector_quantity(key, item)

                    elif len(item.shape) == 1 or len(item.shape) == 2 and item.shape[-1] == 1:
                        item = item.flatten()
                        if not torch.is_floating_point(item):
                            cloud.add_scalar_quantity(f"{key}-id", item, cmap=ID_COLORSCHEME)
                            # if use_colors and 'y' == key:
                            #     y_colors = COLORS[item, :]
                            #     cloud.add_color_quantity(key, y_colors)
                            # else:
                            #     cloud.add_scalar_quantity(key, item, cmap=SCALAR_COLORSCHEME)
                        else:
                            cloud.add_scalar_quantity(key, item, cmap=SCALAR_COLORSCHEME)

                    elif 'color' in key and item.shape[1] == 3:
                        cloud.add_color_quantity(key, item)

                    elif item.shape[-1] > 0 and item.dtype != torch.bool:
                        if item.shape[0] != nodes.num_nodes:
                            item = item[nodes.batch]

                        prediction = item.argmax(dim=-1)
                        probabilities = item.softmax(dim=-1)
                        zeros = (item == 0).all(dim=-1)

                        # Show the prediction, based on the argmax
                        if item.shape[-1] <= COLORS.shape[0] and use_colors:
                            colors = COLORS[:item.shape[-1]]
                            prediction_colors = colors[prediction, :]
                            # label_weights = torch.softmax(x, dim=-1)
                            # weighted_colors = label_weights.unsqueeze(-1) * label_colors.unsqueeze(0)
                            # colors = weighted_colors.sum(dim=1, keepdim=False)
                            probability_colors = (probabilities.unsqueeze(-1) * colors.unsqueeze(0)).sum(dim=1)
                            # zero predictions should be black
                            prediction_colors[zeros, :] = 0
                            probability_colors[zeros, :] = 0
                            cloud.add_color_quantity(f"{key}-max", prediction_colors)
                            cloud.add_color_quantity(f"{key}-prob", probability_colors.numpy(), enabled=True)
                            cloud.add_scalar_quantity(f"{key}-id", prediction, cmap=ID_COLORSCHEME)

                        # cloud.add_scalar_quantity(key, item, enabled=True, cmap='viridis')

                except ValueError as e:
                    print(f"Encountered error while attempting to show node feature '{key}'")
                    print(e)
                    raise e

        def draw_edges(source_nodes: NodeStorage, dest_nodes: NodeStorage, edges: EdgeStorage, prefix='', **kwargs):
            if not hasattr(edges, 'edge_index') or not hasattr(source_nodes, 'pos') or not hasattr(dest_nodes, 'pos'):
                return

            # To draw curves, we need to combine the two point sets
            all_pos = torch.cat([source_nodes.pos, dest_nodes.pos], dim=0)
            remapped_edge_index = edges.edge_index + torch.tensor([0, source_nodes.pos.shape[0]]).unsqueeze(-1)

            curve = polyscope.register_curve_network(
                prefix + '-edges',
                all_pos,
                remapped_edge_index.T,
                radius=0.005 / batch_size,
                enabled=False
            )
            if hasattr(edges, 'edge_weight'):
                curve.add_scalar_quantity('weight', edges.edge_weight, defined_on='edges', enabled=True)

        if isinstance(data, HeteroData):
            show_scale = True
            for scale, item in data.node_items():
                for k, v in data._global_store.items():
                    item[k] = v
                draw_nodes(nodes=item, prefix=scale, enabled=show_scale)
                #show_scale = False  # Only show the first scale, by default
            for (source, _, dest), item in data.edge_items():
                draw_edges(
                    source_nodes=data[source],
                    dest_nodes=data[dest],
                    edges=item,
                    prefix=source if source == dest else f"{source}-to-{dest}",
                )

        else:
            if hasattr(data, 'pos'):
                draw_nodes(data.node_stores[0], prefix='data')
                draw_edges(data.node_stores[0], data.node_stores[0], data.edge_stores[0], prefix='data')

        polyscope.reset_camera_to_home_view()

        # Switch to an overhead view
        position = polyscope.get_view_camera_parameters().get_position()
        position = [position[0] / 2, position[1], position[0] / 2]
        min_pos, max_pos = polyscope.get_bounding_box()
        look_position = (max_pos + min_pos) / 2.0
        polyscope.look_at(position, look_position, fly_to=True)

        polyscope.show()


class DebugShowData(torch.nn.Module):
    def forward(self, data):
        try:
            show_data(data)
        except AttributeError as e:
            print(e)
        return data
