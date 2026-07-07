python step2-get_tier1_event_list.py \
  --neighbors ../data_center/nearest_nerighbors_test.csv \
  --config ./step1-2-config.json \
  --pristine ../relaxed_structures/BP_1014/P/vasprun.xml \
  --out ../data_center/tier1_event_list_angstrom_topo_n2nn.csv \
  --match_check \
  --manifest ../data_center/manifest_test.csv \
  --match_out ../data_center/tier1_event_match_report_angstrom_topo_n2nn.csv
  