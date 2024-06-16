_base_ = '../default_runtime.py'
data_dir = '/home/user/den/waymo/'
category_name = 'Vehicle'
batch_size = 128
point_cloud_range = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5]
box_aware = True

model = dict(
    type='SiamMo',
    backbone=dict(type='VoxelNet',
                  point_cloud_range=point_cloud_range,
                  voxel_size=[0.075, 0.075, 0.15],
                  grid_size=[21, 128, 128],
                  ),
    fuser=dict(type='STFA'),
    head=dict(type='SimpleHead'),
    cfg=dict(
        point_cloud_range=point_cloud_range,
        box_aware=box_aware,
    )
)

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='WaymoDataset',
        path=data_dir,
        category_name=category_name,
        mode='all'
    ),
)

test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)
