#!/bin/bash
LOCAL_EXP_PATH=${pwd}'/..'
IMAGENAME='example_tf_sferes.simg'
DEFNAME='singularity.def'
SANDBOX=true


BLD_ARGS=""
RUN_ARGS=""

while getopts n flag
do
    case $flag in
	
        n)
            echo Nvidia runtime ON
	    RUN_ARGS=$RUN_ARGS" --nv"
            ;;
        ?)
            exit
            ;;
    esac
done

echo "Visualisation available after activating the visu_server.sh script at the http://localhost:6080/"

if $SANDBOX; then
    BLD_ARGS=$BLD_ARGS" --sandbox"
    RUN_ARGS=$RUN_ARGS" -w"
fi
   
if [ -f "$IMAGENAME" ] || [ -d "$IMAGENAME" ]; then
    echo "$IMAGENAME exists"
else
    echo "$IMAGENAME does not exist, building it now from $DEFNAME"
    sudo singularity build --force $BLD_ARGS $IMAGENAME $DEFNAME
fi
sudo singularity shell $RUN_ARGS --bind $(pwd)/../..:/git/sferes2/exp $IMAGENAME
