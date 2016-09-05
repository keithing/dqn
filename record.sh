# example sh record.sh "./videos" "./checkpoints/epoch_70.ckpt"
xvfb-run -s "-screen 0 1400x900x24" python dqn/play.py -d $1 -r -e .01 -w $2
