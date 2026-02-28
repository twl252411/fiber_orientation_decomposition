from abaqus import *
from abaqusConstants import *
from driverUtils import executeOnCaeStartup
import glob
import os

import numpy as np

executeOnCaeStartup()

def discover_input_sets():
    datasets = []

    PROJECT_ROOT = r"D:\github\fiber_orientation_decomposition"  # <- 改成你的主目录（包含 point_angle_files 和 rve_abaqus）
    input_dir = os.path.join(PROJECT_ROOT, "point_angle_files")

    if not os.path.isdir(input_dir):
        raise ValueError("Required folder not found: %s" % input_dir)

    points_pattern = os.path.join(input_dir, "peri_points_*.txt")
    for points_path in sorted(glob.glob(points_pattern)):
        base = os.path.basename(points_path)
        tag = base[len("peri_points_"):-4]
        angles_path = os.path.join(input_dir, "peri_angles_%s.txt" % tag)
        if os.path.isfile(angles_path):
            datasets.append((tag, points_path, angles_path))

    if not datasets:
        raise ValueError(
            "No dataset found in: %s. Expected peri_points_*.txt + peri_angles_*.txt pairs."
            % input_dir
        )
    return datasets


def ensure_2d(data, name):
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.ndim != 2:
        raise ValueError("%s must be a 2D array." % name)
    return data


def create_cylinder_part(model, inc_rad, inc_len):
    s = model.ConstrainedSketch(name="__profile__", sheetSize=6.0)
    s.setPrimaryObject(option=STANDALONE)
    s.CircleByCenterPerimeter(center=(0.0, 0.0), point1=(0.0, inc_rad))
    p = model.Part(name="Part-1", dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p.BaseSolidExtrude(sketch=s, depth=inc_len)
    s.unsetPrimaryObject()
    del model.sketches["__profile__"]
    return p


def create_box_part(model, name, x_size, y_size, z_size):
    s = model.ConstrainedSketch(name="__profile__", sheetSize=200.0)
    s.setPrimaryObject(option=STANDALONE)
    s.rectangle(point1=(0.0, 0.0), point2=(x_size, y_size))
    p = model.Part(name=name, dimensionality=THREE_D, type=DEFORMABLE_BODY)
    p.BaseSolidExtrude(sketch=s, depth=z_size)
    s.unsetPrimaryObject()
    del model.sketches["__profile__"]
    return p


# ---------------------------------------------------------------------------
# Geometry and material parameters
# ---------------------------------------------------------------------------
asp = 10
inc_rad = 2.5
inc_len = inc_rad * asp * 2.0
extra_t = 0.0
rve_size = np.array(
    [inc_len * 2.0 + extra_t, inc_len * 2.0 + extra_t, inc_len * 2.0 + extra_t],
    dtype=float,
)


for tag, points_file, angles_file in discover_input_sets():

    Mdb()

    cae_name = "fiber_rve_%s.cae" % tag
    # Input format compatible with current generator:
    # point_angle_files/peri_points_<tag>.txt
    # point_angle_files/peri_angles_<tag>.txt
    pts = ensure_2d(np.loadtxt(points_file, delimiter=","), "peri_points")
    angs_deg = ensure_2d(np.loadtxt(angles_file, delimiter=","), "peri_angles")

    if pts.shape[0] != angs_deg.shape[0]:
        raise ValueError("points/angles count mismatch for tag %s." % tag)
    if pts.shape[1] != 3:
        raise ValueError("peri_points must have 3 columns.")
    if angs_deg.shape[1] < 2:
        raise ValueError("peri_angles must have at least 2 columns: phi, theta (degree).")

    model = mdb.models["Model-1"]
    a = model.rootAssembly

    # ---------------------------------------------------------------------------
    # Build base fiber part
    # ---------------------------------------------------------------------------
    create_cylinder_part(model, inc_rad, inc_len)

    # ---------------------------------------------------------------------------
    # Instantiate all fibers from points/angles
    # Angle inputs are in degree.
    # ---------------------------------------------------------------------------
    fiber_part = model.parts["Part-1"]
    phi_deg = angs_deg[:, 0]
    theta_deg = angs_deg[:, 1]
    phi_rad = np.deg2rad(phi_deg)
    axis_x = np.sin(phi_rad)
    axis_y = -np.cos(phi_rad)

    instances = []
    for i in range(len(pts)):
        ins_name = "Part-1-%d" % (i + 1)
        a.Instance(name=ins_name, part=fiber_part, dependent=OFF)
        a.translate(instanceList=(ins_name,), vector=(0.0, 0.0, -inc_len / 2.0))
        a.rotate(
            instanceList=(ins_name,),
            axisPoint=(0.0, 0.0, 0.0),
            axisDirection=(0.0, 1.0, 0.0),
            angle=90.0,
        )
        a.rotate(
            instanceList=(ins_name,),
            axisPoint=(0.0, 0.0, 0.0),
            axisDirection=(0.0, 0.0, 1.0),
            angle=float(phi_deg[i]),
        )
        a.rotate(
            instanceList=(ins_name,),
            axisPoint=(0.0, 0.0, 0.0),
            axisDirection=(float(axis_x[i]), float(axis_y[i]), 0.0),
            angle=90.0 - float(theta_deg[i]),
        )
        a.translate(
            instanceList=(ins_name,),
            vector=(float(pts[i, 0]), float(pts[i, 1]), float(pts[i, 2])),
        )
        instances.append(a.instances[ins_name])

    a.InstanceFromBooleanMerge(
        name="Part-New",
        instances=tuple(instances),
        keepIntersections=ON,
        originalInstances=DELETE,
        domain=GEOMETRY,
    )
    a.deleteFeatures(("Part-New-1",))
    model.parts.changeKey(fromName="Part-New", toName="Part-2")
    del model.parts["Part-1"]

    fiber_part = model.parts["Part-2"]
    a.Instance(name="Part-2-1", part=fiber_part, dependent=OFF)
    a.Instance(name="Part-2-2", part=fiber_part, dependent=OFF)

    # ---------------------------------------------------------------------------
    # Create RVE box and crop fibers inside RVE
    # ---------------------------------------------------------------------------
    create_box_part(model, "Part-1", rve_size[0], rve_size[1], rve_size[2])
    box_part = model.parts["Part-1"]
    a.Instance(name="Part-1-1", part=box_part, dependent=OFF)

    a.InstanceFromBooleanCut(
        name="Part-3",
        instanceToBeCut=a.instances["Part-2-1"],
        cuttingInstances=(a.instances["Part-1-1"],),
        originalInstances=DELETE,
    )
    a.InstanceFromBooleanCut(
        name="Part-4",
        instanceToBeCut=a.instances["Part-2-2"],
        cuttingInstances=(a.instances["Part-3-1"],),
        originalInstances=DELETE,
    )
    del model.parts["Part-1"]
    del model.parts["Part-2"]
    del model.parts["Part-3"]
    a.deleteFeatures(("Part-4-1",))
    model.parts.changeKey(fromName="Part-4", toName="Part-2")

    fiber_part = model.parts["Part-2"]
    a.Instance(name="Part-2-1", part=fiber_part, dependent=OFF)
    a.translate(
        instanceList=("Part-2-1",),
        vector=(-rve_size[0] / 2.0, -rve_size[1] / 2.0, -rve_size[2] / 2.0),
    )

    # ---------------------------------------------------------------------------
    # Build matrix block and subtract fibers
    # ---------------------------------------------------------------------------
    create_box_part(
        model,
        "Part-1",
        rve_size[0] + extra_t,
        rve_size[1] + extra_t,
        rve_size[2] + extra_t,
    )
    matrix_part = model.parts["Part-1"]
    a.Instance(name="Part-1-1", part=matrix_part, dependent=OFF)
    a.translate(
        instanceList=("Part-1-1",),
        vector=(
            -(rve_size[0] + extra_t) / 2.0,
            -(rve_size[1] + extra_t) / 2.0,
            -(rve_size[2] + extra_t) / 2.0,
        ),
    )

    a.InstanceFromBooleanCut(
        name="Part-3",
        instanceToBeCut=a.instances["Part-1-1"],
        cuttingInstances=(a.instances["Part-2-1"],),
        originalInstances=DELETE,
    )
    del model.parts["Part-1"]
    a.deleteFeatures(("Part-3-1",))
    model.parts.changeKey(fromName="Part-3", toName="Part-1")

    # ---------------------------------------------------------------------------
    # Merge matrix + fibers into final part
    # ---------------------------------------------------------------------------
    matrix_part = model.parts["Part-1"]
    a.Instance(name="Part-1-1", part=matrix_part, dependent=OFF)
    fiber_part = model.parts["Part-2"]
    a.Instance(name="Part-2-1", part=fiber_part, dependent=OFF)
    a.translate(
        instanceList=("Part-2-1",),
        vector=(-rve_size[0] / 2.0, -rve_size[1] / 2.0, -rve_size[2] / 2.0),
    )

    volume1 = matrix_part.getMassProperties()["volume"]
    volume2 = fiber_part.getMassProperties()["volume"]

    print("Tag %s: particle volume fraction = %s" % (tag, volume2 / (volume2 + volume1)))

    # ---------------------------------------------------------------------------
    # Save CAE
    # ---------------------------------------------------------------------------
    mdb.saveAs(pathName=cae_name)
