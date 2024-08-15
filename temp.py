

    traj_file = os.path.join(path, 'traj.txt')
    with open(traj_file, 'r') as poses_file:
        poses = poses_file.readlines()
    
    height = 680
    width = 1200
    fx = 600
    fy = 600
    cx = 599.5
    cy = 339.5

    FovY = focal2fov(fy, height)
    FovX = focal2fov(fx, width)
    
    image_paths = sorted(glob.glob(os.path.join(path, 'images/*')))
    depth_paths = sorted(glob.glob(os.path.join(path, 'depths/*')))
    
    cam_infos = []
    test_cam_infos = []
    mat_list=[]
    viz_list=[]
    pc_init = np.zeros((0,3))
    color_init = np.zeros((0,3))

for idx, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
        mat = np.array(poses[idx].split('\n')[0].split(' ')).reshape((4,4)).astype('float64')
        mat_list.append(mat)

        R = mat[:3,:3]
        T = mat[:3, 3]

        R_gt=R.copy()
        T_gt=T.copy()

        # Invert
        T = -R.T @ T # convert from real world to GS format: R=R, T=T.inv()
        T_gt = -R_gt.T @ T_gt # convert from real world to GS format: R=R, T=T.inv()

        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        depth = Image.open(depth_path)
        depth_scaled = Image.fromarray(np.array(depth) * 255.0)
        
        if len(single_frame_id)>0 and (idx not in single_frame_id):
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, Cy=cy, Cx=cx, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height, depth=depth_scaled, R_gt=R_gt, T_gt=T_gt,
                                mat=mat, raw_pc=None, kdtree=None)
            test_cam_infos.append(cam_info)
        else:
            o3d_depth = o3d.geometry.Image(np.array(depth).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(image).astype(np.uint8))
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600, 600, 599.5, 339.5)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=6553.5, depth_trunc=1000, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            o3d_pc = o3d_pc.transform(mat)
            raw_pc = np.asarray(o3d_pc.points)
            kdtree = KDTree(raw_pc)
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, Cy=cy, Cx=cx, image=image,
                                image_path=image_path, image_name=image_name, width=width, height=height, depth=depth_scaled, R_gt=R_gt, T_gt=T_gt,
                                mat=mat, # Poses are w.r.t. the original frame
                                raw_pc=raw_pc,
                                kdtree=kdtree
                                )
            cam_infos.append(cam_info)
    
    train_cam_infos = cam_infos

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")

    if not load_ply:
        for idx, (image_path, depth_path) in enumerate(zip(image_paths, depth_paths)):
            if len(single_frame_id)>0 and (idx not in single_frame_id):
                continue
            mat = np.array(poses[idx].split('\n')[0].split(' ')).reshape((4,4)).astype('float64')
            image = Image.open(image_path)
            depth = Image.open(depth_path)
            o3d_depth = o3d.geometry.Image(np.array(depth).astype(np.float32))
            o3d_image = o3d.geometry.Image(np.array(image).astype(np.uint8))
            # Replica dataset cam: H: 680 W: 1200 fx: 600.0 fy: 600.0 cx: 599.5 cy: 339.5 png_depth_scale: 6553.5
            o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(1200, 680, 600, 600, 599.5, 339.5)
            # o3d_pc = o3d.geometry.PointCloud.create_from_depth_image(depth=o3d_depth, intrinsic=o3d_intrinsic, extrinsic=np.identity(4), depth_scale=6553.5, stride=50)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(o3d_image, o3d_depth, depth_scale=6553.5, depth_trunc=1000, convert_rgb_to_intensity=False)
            o3d_pc = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd_img, intrinsic=o3d_intrinsic, extrinsic=np.identity(4))
            dist = np.linalg.norm(np.asarray(o3d_pc.points), axis=1)

            o3d_pc = o3d_pc.transform(mat)
            pc_init = np.concatenate((pc_init, np.asarray(o3d_pc.points)), axis=0)
            color_init = np.concatenate((color_init, np.asarray(o3d_pc.colors)), axis=0)
        
        num_pts = pc_init.shape[0]
        xyz = pc_init
        # color_init = np.ones_like(color_init) # !!! initialize all color to white for viz
        pcd = BasicPointCloud(points=xyz, colors=color_init, normals=np.zeros((num_pts, 3)))
        storePly(ply_path, pc_init, color_init*255)
        print('save pcd')
    try:
        pcd = fetchPly(ply_path)
        print('read: ', pcd.points.shape)
    except:
        pcd = None
    
    gaussian_init = None
    if init_w_gaussian:
        mean_xyz, mean_rgb, cov = precompute_gaussians(torch.tensor(pcd.points).to('cuda'), torch.tensor(pcd.colors).to('cuda'), voxel_size)
        gaussian_init={"mean_xyz": mean_xyz, "mean_rgb": mean_rgb, "cov": cov}
    else:
        if voxel_size is not None:
            # downsample
            o3d_pcd = o3d.geometry.PointCloud()
            o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
            o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size)
            pc_init = np.asarray(o3d_pcd.points)
            color_init = np.asarray(o3d_pcd.colors)
            pcd = BasicPointCloud(points=pc_init, colors=color_init, normals=np.zeros((pc_init.shape[0], 3)))

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           pseudo_cameras=None,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           gaussian_init=gaussian_init)
    return scene_info