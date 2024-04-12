brightness =[0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
gauss = [0.005]

from itertools import product

args = list(product(brightness, gauss))
print(len(args))

for brightness, gauss in args:
    outdir = f'~/publication/VIT_openframe_newaug_brightness_{brightness}_gauss_{gauss}'
    print(f'OUTDIR={outdir} BRIGHTNESS={brightness} GAUSS={gauss} bash gen_all_openframe_grid.sh;')