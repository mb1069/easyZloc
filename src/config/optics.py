voxel_sizes = (100, 85, 85)

target_psf_shape = (41, 40, 40)

model_kwargs = dict(
    wl=660,
    na=1.3,
    ni=1.51,
    res=voxel_sizes[1],
    size=target_psf_shape[1],
    zsize=target_psf_shape[0],
    zres=voxel_sizes[0],
    vec_corr="none",
    condition="none",
)
