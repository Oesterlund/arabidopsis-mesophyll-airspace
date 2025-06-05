Adjust `image_dir` path in  `segment_semantic.sh`. Everything else should fine.

Run in the following order
 1. `segment_semantic.sh`
 2. `post_process_semantic_seg.sh`
 3. `segment_instance.sh`
 4. `post_process_instance_seg.sh`
 
Use `viewer.py` to inspect results.
