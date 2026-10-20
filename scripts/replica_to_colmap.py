"""
Convert a NICE-SLAM-style Replica sequence into the COLMAP layout that
`train.py` reads (images/ + sparse/0/).

Replica ships ground-truth poses, so the camera extrinsics are copied straight
across.  The sparse point cloud, on the other hand, is triangulated from SIFT
matches on the rendered images and NOT from the ground-truth depth maps: the
billboards' initial positions (and therefore `--rotation_init pca`'s initial
normals) must not be seeded with the geometry we later evaluate against.

  python scripts/replica_to_colmap.py --replica_root <.../Replica> \
      --out_root <.../replica_colmap> --scenes room0 office0 --stride 10

traj.txt holds camera-to-world matrices in OpenCV convention (x right, y down,
z forward) with the world z-up -- the same convention COLMAP uses, so no axis
flip is applied.  (NICE-SLAM's own loader negates columns 1 and 2 because it
wants OpenGL.)  Verified on room0: R orthonormal to 4e-16, det 1, and the mean
camera up -R[:,1] comes out as world +z.
"""
import argparse, json, os, shutil
import numpy as np


def load_intrinsics(replica_root):
    with open(os.path.join(replica_root, "cam_params.json")) as f:
        return json.load(f)["camera"]


def convert_scene(replica_root, out_root, scene, stride, link):
    """Lay out images/ and return {image name: world-to-camera 3x4}."""
    src = os.path.join(replica_root, scene)
    res = os.path.join(src, "results")
    if not os.path.isdir(res):
        raise SystemExit(f"missing {res}")
    frames = sorted(f for f in os.listdir(res) if f.startswith("frame") and f.endswith(".jpg"))
    traj = np.loadtxt(os.path.join(src, "traj.txt")).reshape(-1, 4, 4)
    if len(traj) != len(frames):
        raise SystemExit(f"{scene}: {len(frames)} frames but {len(traj)} poses")
    sel = list(range(0, len(frames), stride))

    dst = os.path.join(out_root, scene)
    img_dir = os.path.join(dst, "images")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(os.path.join(dst, "sparse", "0"), exist_ok=True)

    poses = {}
    for i in sel:
        name = frames[i]
        tgt, s = os.path.join(img_dir, name), os.path.join(res, name)
        if not os.path.exists(tgt):
            os.symlink(s, tgt) if link else shutil.copy2(s, tgt)
        c2w = traj[i]
        Rcw = c2w[:3, :3].T
        poses[name] = np.hstack([Rcw, (-Rcw @ c2w[:3, 3]).reshape(3, 1)])

    cam = load_intrinsics(replica_root)
    with open(os.path.join(dst, "replica_meta.json"), "w") as f:
        json.dump({"scene": scene, "stride": stride,
                   "frames": [frames[i] for i in sel], "frame_indices": sel,
                   "depth_scale": cam.get("scale"), "intrinsics": cam,
                   "source": src}, f, indent=1)
    print(f"{scene}: {len(sel)} of {len(frames)} frames -> {dst}")
    return dst, poses


def triangulate(dst, poses, cam, use_gpu=True, reuse_db=False,
                peak_threshold=0.0066, max_features=8192):
    """
    SIFT + exhaustive matching, then point_triangulator against the known poses.

    The reconstruction handed to the triangulator is assembled from the
    database's own rigs, frames, cameras and images rather than from a text
    model we write ourselves.  COLMAP 3.12 keys frames to database image ids,
    and `extract_features` does not number images in filename order, so a
    hand-written model disagrees with the database on both image ids and frame
    data ids and the triangulator rejects it.
    """
    import pycolmap
    db_path = os.path.join(dst, "database.db")
    imgs_dir = os.path.join(dst, "images")
    sp = os.path.join(dst, "sparse", "0")

    if not (reuse_db and os.path.exists(db_path)):
        if os.path.exists(db_path):
            os.remove(db_path)
        dev = pycolmap.Device.auto if use_gpu else pycolmap.Device.cpu
        # Replica gives exact intrinsics; pin them instead of letting SIFT guess,
        # and pin PINHOLE so the model matches what we build below.
        ro = pycolmap.ImageReaderOptions()
        ro.camera_model = "PINHOLE"
        ro.camera_params = f"{cam['fx']},{cam['fy']},{cam['cx']},{cam['cy']}"
        eo = pycolmap.FeatureExtractionOptions()
        # Replica's office scenes are largely plain paint, so SIFT at its
        # defaults returns very few keypoints and the triangulated cloud comes
        # out too sparse to initialise MCMC: room1 started from 47k points and
        # trained fine, office1 started from 7.9k, had to be cloned 25x to
        # reach cap_max, and diverged (alpha collapsed to zero by iter 7000).
        # Lowering the peak threshold and raising the cap recovers features on
        # low-contrast surfaces.
        eo.sift.peak_threshold = peak_threshold
        eo.sift.max_num_features = max_features
        pycolmap.extract_features(db_path, imgs_dir, reader_options=ro,
                                  extraction_options=eo,
                                  camera_mode=pycolmap.CameraMode.SINGLE, device=dev)
        pycolmap.match_exhaustive(db_path, device=dev)

    db = pycolmap.Database.open(db_path)
    db_imgs = db.read_all_images()
    db_frames = db.read_all_frames()
    db_rigs = db.read_all_rigs()
    db_cams = db.read_all_cameras()
    id2name = {im.image_id: im.name for im in db_imgs}

    missing = [n for n in id2name.values() if n not in poses]
    if missing:
        raise SystemExit(f"no pose for {len(missing)} database images, e.g. {missing[:3]}")

    for fr in db_frames:
        ids = list(fr.image_ids)
        if len(ids) != 1:
            raise SystemExit(f"frame {fr.frame_id} holds {len(ids)} images; "
                             "this converter assumes one camera per frame")
        # trivial rig: sensor_from_rig is identity, so rig_from_world == cam_from_world
        fr.rig_from_world = pycolmap.Rigid3d(poses[id2name[ids[0].id]])

    rec = pycolmap.Reconstruction()
    for c in db_cams:          # cameras must exist before the rigs that use them
        rec.add_camera(c)
    rec.set_rigs_and_frames(db_rigs, db_frames)
    for im in db_imgs:
        rec.add_image(im)
    for fr in db_frames:
        rec.register_frame(fr.frame_id)
    db.close()
    print(f"  input model: {rec.num_images()} images, {rec.num_cameras()} camera(s), "
          f"{rec.num_reg_frames()} registered frames")

    pycolmap.triangulate_points(rec, db_path, imgs_dir, sp, clear_points=True)
    out = pycolmap.Reconstruction(sp)
    n = out.num_points3D()
    print(f"  triangulated {n:,} points over {out.num_reg_images()} images; "
          f"mean reprojection error {out.compute_mean_reprojection_error():.3f} px, "
          f"mean track length {out.compute_mean_track_length():.2f}")
    if n < 1000:
        print("  WARNING: very few points; training will start from a poor initialisation")
    return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--replica_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlinking")
    ap.add_argument("--no_gpu", action="store_true")
    ap.add_argument("--reuse_db", action="store_true",
                    help="keep an existing database.db (skips feature extraction)")
    ap.add_argument("--peak_threshold", type=float, default=0.0066,
                    help="SIFT peak threshold; COLMAP's default is 0.0066 and "
                         "lowering it finds features on plain surfaces")
    ap.add_argument("--max_features", type=int, default=8192)
    ap.add_argument("--skip_triangulation", action="store_true")
    a = ap.parse_args()
    cam = load_intrinsics(a.replica_root)
    for s in a.scenes:
        dst, poses = convert_scene(a.replica_root, a.out_root, s, a.stride, not a.copy)
        if not a.skip_triangulation:
            triangulate(dst, poses, cam, not a.no_gpu, a.reuse_db,
                        a.peak_threshold, a.max_features)
