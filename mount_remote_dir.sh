#!/bin/bash

USER_CLIENT="shubhadityaburela"
NODE="node748.math.priv10.lan"

USER_NODE="burela"
PROJECT="OCsPOD"


LOCAL_PROJECT_PATH="/Users/${USER_CLIENT}/Python/${PROJECT}/"
REMOTE_KERNEL_PATH="/net/homes/math/${USER_NODE}/sshfs_mounts/${PROJECT}/"



cmd0="cd ${REMOTE_KERNEL_PATH}; echo \"Mounting local files ( PW of the current PC )\";"
cmd1="sshfs ${USER_CLIENT}@localhost:\"${LOCAL_PROJECT_PATH}\" \"${REMOTE_KERNEL_PATH}\$i\" -p 2224 ;"
#cmd1="ssh ${USER_CLIENT}@${CLIENT}"
cmd2="mkdir \"${REMOTE_KERNEL_PATH}\$i\"; rm current.txt; echo \"\$i\" &> current.txt;"
cmd3="read -p \"Press any key to unmount\"; echo \"Closed\"; killall sshfs; rm -rf \$i;"
cmd4="i=1; occ=\$(ls); while [[ \"\$occ\" == *\"\$i\"* ]]; do i=\$((i+1)); done; echo \$i; echo \$occ;"

echo "Make sure the VPN is connected and resources are allocated" 
echo "Login to the node ( cluster PW )"
ssh -R 2224:localhost:22 ${USER_NODE}@${NODE} -t $cmd0 $cmd4 $cmd2 $cmd1 $cmd3



#ssh ${USER_NODE}@${ENTRY_SERVER} -t "salloc -p gbr --gres=gpu:${GPU_NAME}:${NUM_GPUS} --mem=${MEM};  cmd"



