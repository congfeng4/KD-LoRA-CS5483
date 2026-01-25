rsync -avz --progress \
      -e "ssh -p 22" \
    GPU25:/mnt/data2/congfeng/kd-lora/dataset.zip \
     /home/user/fc/kd-lora

rsync -avz --progress \
      -e "ssh -p 22" \
    GPU25:/mnt/data2/congfeng/kd-lora/models.zip \
     /home/user/fc/kd-lora

rsync -avz --progress \
      -e "ssh -p 22" \
    GPU25:/mnt/data2/congfeng/kd-lora/results \
     /home/user/fc/kd-lora \
     --exclude ckpt --exclude checkpoint

rsync -avz --progress \
      -e "ssh -p 22" \
      /home/user/fc/kd-lora/models \
    GPU25:/mnt/data2/congfeng/kd-lora/

