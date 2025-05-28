
1. create zarr dataset by running:
  ```bash
  python process_expert_data.py
  ```

  if you want to calb for the table, run:
  ```bash
  python scripts/gen_table_pcd.py
  python scripts/process_expert_data.py
  ```
  
2. Train the policy. For example:
  ```bash
    bash scripts/train_policy.sh idp3 g1_sim_place_expert test 0 0
  ```
  remember to change idp3.yaml to modify your requirements.