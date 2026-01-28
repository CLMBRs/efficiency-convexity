from skimage import color
import open3d as o3d
import numpy as np
import pandas as pd
import sys


def create_grid_lines(
    min_bound: np.ndarray,
    max_bound: np.ndarray,
    step: int = 10,
    plane: str = "xy",
    color: tuple[float, float, float] = (0.6, 0.6, 0.6),
    lo: bool = True,
) -> o3d.geometry.LineSet:
    """
    Returns grid lines which are aligned to a specific plane. These grid lines are also sized and placed to be behind and only as big as the points

    Args:
        min_bound (np.ndarray): The smallest x, y, and z values
        max_bound (np.ndarray): The largest x, y, and z values
        step (int): The distance between grid lines
        plane (str): The name of the plane to draw (xy, xz, or yz)
        color (tuple[float, float, float]): The color to draw the lines (rgb from 0 to 1)
        lo (bool): Whether to draw the grid aligned to the min bound or the max bound

    Returns:
        o3d.geometry.LineSet: The grid which can be added to the Visualizer
    """
    lines = []
    points = []

    low_mov = 0 if lo else step
    hi_mov = step if lo else 0
    x_vals = np.arange(min_bound[0] - low_mov, max_bound[0] + hi_mov, step)
    y_vals = np.arange(min_bound[1] - low_mov, max_bound[1] + hi_mov, step)
    z_vals = np.arange(min_bound[2] - low_mov, max_bound[2] + hi_mov, step)

    if plane == "xy":
        z = min_bound[2] if lo else max_bound[2]  # fixed
        for x in x_vals:
            points.append([x, y_vals[0], z])
            points.append([x, y_vals[-1], z])
            lines.append([len(points) - 2, len(points) - 1])
        for y in y_vals:
            points.append([x_vals[0], y, z])
            points.append([x_vals[-1], y, z])
            lines.append([len(points) - 2, len(points) - 1])

    elif plane == "xz":
        y = min_bound[1] if lo else max_bound[1]  # fixed
        for x in x_vals:
            points.append([x, y, z_vals[0]])
            points.append([x, y, z_vals[-1]])
            lines.append([len(points) - 2, len(points) - 1])
        for z in z_vals:
            points.append([x_vals[0], y, z])
            points.append([x_vals[-1], y, z])
            lines.append([len(points) - 2, len(points) - 1])

    elif plane == "yz":
        x = min_bound[0] if lo else max_bound[0]  # fixed
        for y in y_vals:
            points.append([x, y, z_vals[0]])
            points.append([x, y, z_vals[-1]])
            lines.append([len(points) - 2, len(points) - 1])
        for z in z_vals:
            points.append([x, y_vals[0], z])
            points.append([x, y_vals[-1], z])
            lines.append([len(points) - 2, len(points) - 1])

    grid = o3d.geometry.LineSet()
    grid.points = o3d.utility.Vector3dVector(points)
    grid.lines = o3d.utility.Vector2iVector(lines)
    grid.colors = o3d.utility.Vector3dVector([color for _ in lines])
    return grid


def custom_draw(
    pcd: o3d.geometry.PointCloud,
    extras: list[o3d.geometry.LineSet] = None,
    point_size: float = 15.0,
    camera_position: np.ndarray = None,
    lookat: np.ndarray = None,
    up: np.ndarray = None,
):
    """
    Draws the point cloud and takes a screenshot which is then saved, then closes the window

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud which will be drawn
        extras (list[o3d.geometry.LineSet]): The list of extra objects to draw, such as grid lines
        point_size (float): The size of the points in the point cloud
        camera_position (np.ndarray): The position of the camera
        lookat (np.ndarray): Where the camera should be facing
        up (np.ndarray): The up direction of the camera
    """
    extras = [] if extras is None else extras

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600)
    vis.add_geometry(pcd)
    for geo in extras:
        vis.add_geometry(geo)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([1, 1, 1])  # white background
    vis.poll_events()
    vis.update_renderer()

    # Set camera parameters (position, lookat, up)
    if camera_position is not None and lookat is not None and up is not None:
        ctr = vis.get_view_control()
        ctr.set_lookat(lookat)
        ctr.set_front((np.array(lookat) - np.array(camera_position)).tolist())
        ctr.set_up(up)
        ctr.set_zoom(1.1)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(f"./colors/output/color/{sys.argv[1]}.png")
    # vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    cielab_df = pd.read_csv("./colors/data/cnum-vhcm-lab-new.txt", sep="\t", header=0)
    cielab_df.columns = ["chip_id", "V", "H", "C", "m_hue", "m_value", "L", "A", "B"]
    cielab_df.sort_values(by="chip_id", inplace=True)
    raw_points = cielab_df[["L", "A", "B"]].values
    points = np.array(raw_points)

    rot = 0 if len(sys.argv) <= 2 else int(sys.argv[2])
    show_color_diff = len(sys.argv) == 4 and sys.argv[3] == "diff"
    if rot > 0:
        colors = []
        for _, row in cielab_df.iterrows():
            if int(row["H"]) == 0:
                colors.append(row[["L", "A", "B"]].values)
                continue
            h_val = (int(row["H"]) + rot - 1) % 40 + 1
            new_color = cielab_df.loc[
                (cielab_df["H"] == h_val) & (cielab_df["V"] == row["V"])
            ].iloc[0]
            colors.append(new_color[["L", "A", "B"]].values)
        colors = np.array(colors).astype(np.float64)
        if not show_color_diff:
            colors = color.lab2rgb(colors)
    else:
        colors = color.lab2rgb(points)

    if show_color_diff:
        colors = np.sqrt(np.sum(np.square(points - colors), axis=1))
        colors /= np.max(colors)
        colors = np.tile(colors, (3, 1)).T

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=200.0, origin=[0, 0, 0]
    )

    # Determine bounding box for grid placement
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)

    # Create grids on all three major planes
    xy_grid = create_grid_lines(
        min_bound, max_bound, step=10, plane="xy", color=[0, 0, 0]
    )
    xz_grid = create_grid_lines(
        min_bound, max_bound, step=10, plane="xz", color=[0, 0, 0]
    )
    yz_grid = create_grid_lines(
        min_bound, max_bound, step=10, plane="yz", color=[0, 0, 0]
    )

    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()
    custom_draw(
        pcd,
        extras=[xy_grid, xz_grid, yz_grid],
        lookat=min_bound + np.array([-20, 0, 10]),
        up=np.array([1, 0, 0]),
        camera_position=center + np.array([-135, -180, -220]),
    )
