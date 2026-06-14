# Graph Report - .  (2026-06-09)

## Corpus Check
- Corpus is ~44,103 words - fits in a single context window. You may not need a graph.

## Summary
- 744 nodes · 1150 edges · 49 communities (42 shown, 7 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 72 edges (avg confidence: 0.71)
- Token cost: 4,200 input · 1,850 output

## Community Hubs (Navigation)
- [[_COMMUNITY_GNN Dataset Pipeline|GNN Dataset Pipeline]]
- [[_COMMUNITY_YAML Security Feature Extraction|YAML Security Feature Extraction]]
- [[_COMMUNITY_Graph Data Structures|Graph Data Structures]]
- [[_COMMUNITY_Model Config & Evaluation|Model Config & Evaluation]]
- [[_COMMUNITY_Test Metrics & Results|Test Metrics & Results]]
- [[_COMMUNITY_System Architecture (Thesis)|System Architecture (Thesis)]]
- [[_COMMUNITY_Cluster Graph Builder|Cluster Graph Builder]]
- [[_COMMUNITY_RF Feature Importances|RF Feature Importances]]
- [[_COMMUNITY_Inference Pipeline|Inference Pipeline]]
- [[_COMMUNITY_GA Optimisation Results|GA Optimisation Results]]
- [[_COMMUNITY_kubescan CLI & Ensemble|kubescan CLI & Ensemble]]
- [[_COMMUNITY_Manifest Data Acquisition|Manifest Data Acquisition]]
- [[_COMMUNITY_GAT Model Training|GAT Model Training]]
- [[_COMMUNITY_GA Ensemble Weights|GA Ensemble Weights]]
- [[_COMMUNITY_Cross-Validation Results|Cross-Validation Results]]
- [[_COMMUNITY_YAML Parser Tests|YAML Parser Tests]]
- [[_COMMUNITY_Security Tools Scanner|Security Tools Scanner]]
- [[_COMMUNITY_Graph Augmentation|Graph Augmentation]]
- [[_COMMUNITY_RF Classifier Inference|RF Classifier Inference]]
- [[_COMMUNITY_RF Dataset Builder|RF Dataset Builder]]
- [[_COMMUNITY_RF Dataset Enrichment|RF Dataset Enrichment]]
- [[_COMMUNITY_Cross-Val Metrics|Cross-Val Metrics]]
- [[_COMMUNITY_Error Handling|Error Handling]]
- [[_COMMUNITY_RF Training Results|RF Training Results]]
- [[_COMMUNITY_Escape Signal Computation|Escape Signal Computation]]
- [[_COMMUNITY_RF Model Training|RF Model Training]]
- [[_COMMUNITY_HOSTPATH Data Fix|HOSTPATH Data Fix]]
- [[_COMMUNITY_Attack Repo Ingestion|Attack Repo Ingestion]]
- [[_COMMUNITY_RF Test Metrics|RF Test Metrics]]
- [[_COMMUNITY_RF Hyperparameters|RF Hyperparameters]]
- [[_COMMUNITY_Ensemble Checkpoint Loading|Ensemble Checkpoint Loading]]
- [[_COMMUNITY_Graph Builder Tests|Graph Builder Tests]]
- [[_COMMUNITY_Data Splits (5-Fold CV)|Data Splits (5-Fold CV)]]
- [[_COMMUNITY_Test Fixtures|Test Fixtures]]
- [[_COMMUNITY_CLI Integration Tests|CLI Integration Tests]]
- [[_COMMUNITY_CLI Entry Point|CLI Entry Point]]
- [[_COMMUNITY_Package Init|Package Init]]
- [[_COMMUNITY_Checkpoint Resolution|Checkpoint Resolution]]
- [[_COMMUNITY_CLI Usage Docs|CLI Usage Docs]]
- [[_COMMUNITY_Data Layout Docs|Data Layout Docs]]
- [[_COMMUNITY_Coding Standards|Coding Standards]]
- [[_COMMUNITY_Repository Structure|Repository Structure]]
- [[_COMMUNITY_Thesis Revision Log|Thesis Revision Log]]

## God Nodes (most connected - your core abstractions)
1. `feature_importances` - 26 edges
2. `extract_features_from_resource()` - 24 edges
3. `KubeClusterDataset` - 23 edges
4. `_run_inference_pipeline()` - 16 edges
5. `best_weights` - 16 edges
6. `KubeGAT` - 16 edges
7. `KubescanError` - 14 edges
8. `build_cluster_graph()` - 14 edges
9. `KubeGAT` - 13 edges
10. `Path` - 13 edges

## Surprising Connections (you probably didn't know these)
- `Multi-Hop Attack Chain (pod-escape -> lateral movement -> impact)` --semantically_similar_to--> `Graph Edge Types (0=directory_proximity, 1=privilege_reach, 2=sa_lateral, 3=semantic_namespace, 4=rbac_priv)`  [INFERRED] [semantically similar]
  README.md → thesis/memory/01_dataset.md
- `run_gnn_ensemble()` --calls--> `DataLoader`  [INFERRED]
  kubescan/src/kubescan/model/ga_ensemble.py → research/models/train_gnn.py
- `Layer 1: Random Forest Classifier` --shares_data_with--> `Node Feature Vector (26-dim: indices 0-24 binary flags + risk_score at index 25)`  [INFERRED]
  CLAUDE.md → thesis/memory/01_dataset.md
- `Multi-Hop Attack Chain (pod-escape -> lateral movement -> impact)` --conceptually_related_to--> `Layer 2: Graph Attention Network (GAT) 5-fold Ensemble`  [EXTRACTED]
  README.md → CLAUDE.md
- `RF-Only P@5=0.20 (GNN contribution is critical: 4x improvement)` --conceptually_related_to--> `Layer 2: Graph Attention Network (GAT) 5-fold Ensemble`  [EXTRACTED]
  thesis/memory/06_thesis_narrative.md → CLAUDE.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Three-Layer Ensemble: RF -> GAT -> GA Scorer** — tfe_claude_layer1_rf, tfe_claude_layer2_gat, tfe_claude_layer3_ga_ensemble [EXTRACTED 1.00]
- **Dataset Sources Composing rf_dataset.csv** — thesis_memory_01_dataset_rahman, thesis_memory_01_dataset_badpods, thesis_memory_01_dataset_kubernetes_goat, thesis_memory_01_dataset_rf_dataset [EXTRACTED 1.00]
- **kubescan Inference Module Trio (RF + GAT + GA)** — tfe_claude_rf_classifier, tfe_claude_gat_encoder, tfe_claude_ga_ensemble_module [EXTRACTED 1.00]
- **Three-Layer kubescan Model Pipeline: RF → Cluster Graph → GAT → GA Ensemble** — figures_fig_arquitectura_layer1_rf, figures_fig_arquitectura_cluster_graph, figures_fig_arquitectura_layer2_gat, figures_fig_arquitectura_layer3_ga_ensemble [EXTRACTED 1.00]
- **Evaluation Results: RF confusion, GNN confusion, GNN evolution, Ensemble P@k** — figures_fig_rf_confusion_matrix, figures_fig_gnn_confusion_matrix, figures_fig_gnn_evolution_training_curve, figures_fig_ensemble_pat_k_precision_chart [INFERRED 0.95]
- **Thesis Document Artifacts: Research doc, UNIR instructions, figures** — pdf_kubernetes_security_ml_document, pdf_instrucciones_tfe_guide, latex_logo_unir_institution [INFERRED 0.85]

## Communities (49 total, 7 thin omitted)

### Community 0 - "GNN Dataset Pipeline"
Cohesion: 0.05
Nodes (52): compute_feature_stats(), KubeClusterDataset, load_split(), npz_to_data(), gnn_dataset.py ============== PyTorch Geometric dataset wrapper for the K8s clus, In-memory PyG dataset of Kubernetes cluster graphs.      Parameters     --------, Load all matching .npz files into memory., Load a subset of graphs specified by a split .txt file (one cluster per line). (+44 more)

### Community 1 - "YAML Security Feature Extraction"
Cohesion: 0.05
Nodes (61): _check_allow_privi_escalation(), _check_capabilities(), _check_docker_sock(), _check_host_aliases(), _check_host_flags(), _check_hostpath_mount(), _check_image_uses_latest(), _check_insecure_http() (+53 more)

### Community 2 - "Graph Data Structures"
Cohesion: 0.08
Nodes (48): GraphResult, IntEnum, Data, DiGraph, Path, Path, _add_lateral_edges(), _add_privilege_edges() (+40 more)

### Community 3 - "Model Config & Evaluation"
Cohesion: 0.05
Nodes (50): Ensemble Score Formula: score = w_rf*risk + w_gnn*chain + w_esc*escape, GA Config YAML (generations=150, pop_size=60, alpha=0.7, beta=0.3), GNN Config YAML (hidden=64, heads=4, layers=3, lr=5e-4, epochs=300), RF Config YAML (test_size=0.20, seed=42), evaluate_test_set.py - Held-out Test Evaluation Script, Research Training Pipeline (8-step sequence), run_ga_ensemble.py - GA Ensemble Weight Optimisation Script, train_gnn.py - GNN Training Script (5-fold CV) (+42 more)

### Community 4 - "Test Metrics & Results"
Cohesion: 0.06
Nodes (31): classification_metrics, accuracy, confusion_matrix, macro_f1, per_class_f1, 0, 1, 2 (+23 more)

### Community 5 - "System Architecture (Thesis)"
Cohesion: 0.09
Nodes (31): Cluster Graph G=(V,E) with 5 edge types (dir_proximity, privilege_reach, sa_lateral, co_namespace, RBAC_priv), Layer 1: Random Forest (500 trees, 25 features, risk_score output), Layer 2: GAT (3 layers, 4 heads, mean+max pooling, p_chain output), Layer 3: Genetic Algorithm Ensemble (final classification), Output Classification: CLEAN / ISOLATED / ATTACK_CHAIN, kubescan 3-Layer System Architecture Diagram, YAML Manifests Input (local directory or kubectl), Precision@5 = 0.80 (exceeds objective of 0.70) (+23 more)

### Community 6 - "Cluster Graph Builder"
Cohesion: 0.11
Nodes (29): build_cluster_graph(), build_lookups(), _compute_graph_label(), dir_key(), _extract_repo_relpath(), _get_pod_spec(), graph_to_arrays(), graph_to_json() (+21 more)

### Community 7 - "RF Feature Importances"
Cohesion: 0.08
Nodes (26): feature_importances, all_secrets, ALLOW_PRIVI, cap_misuse, CAP_SYS_ADMIN, CAP_SYS_MODULE, DOCKERSOCK_PATH, HOST_ALIAS (+18 more)

### Community 8 - "Inference Pipeline"
Cohesion: 0.14
Nodes (24): build_graph(), _build_node_feature_vector(), _build_rf_input(), compute_ensemble_score(), extract_cluster_features(), _flag_summary(), graph_to_pyg(), main() (+16 more)

### Community 9 - "GA Optimisation Results"
Cohesion: 0.08
Nodes (23): best_weights, alpha, beta, fpr_clean, grid_best, k, mode, note (+15 more)

### Community 10 - "kubescan CLI & Ensemble"
Cohesion: 0.18
Nodes (23): EnsembleScorer, _expand_yaml_to_files(), _fetch_live_manifests(), _flag_list(), _format_node_row(), _format_verdict_line(), live(), _print_node_table() (+15 more)

### Community 11 - "Manifest Data Acquisition"
Cohesion: 0.19
Nodes (15): build_github_tasks(), build_gitlab_tasks(), github_blob_to_raw(), gitlab_blob_to_raw(), log(), main(), download_github_manifests.py ============================= Download the actual K, Convert a GitLab blob URL to a raw file URL.      Input:  https://gitlab.com/own (+7 more)

### Community 12 - "GAT Model Training"
Cohesion: 0.14
Nodes (16): KubeGAT, Data, device, device, Path, Tensor, ga_ensemble.py ============== Ensemble scorer.  Combines RF risk scores, GNN cha, Average softmax probabilities across all fold models.     Returns (chain_prob, c (+8 more)

### Community 13 - "GA Ensemble Weights"
Cohesion: 0.10
Nodes (19): alpha, beta, fpr_clean, grid_best, p_at_k, w_escape, w_gnn, w_rf (+11 more)

### Community 14 - "Cross-Validation Results"
Cohesion: 0.11
Nodes (18): accuracy_mean, accuracy_std, binary, fold_f1s, fold_p5s, macro_f1_mean, macro_f1_std, model_config (+10 more)

### Community 15 - "YAML Parser Tests"
Cohesion: 0.23
Nodes (15): Path, test_yaml_parser.py =================== Unit tests for kubescan/utils/yaml_parse, test_extract_cluster_features_returns_both_manifests(), test_extract_features_attack_cap_sys_admin_set(), test_extract_features_attack_docker_sock_set(), test_extract_features_attack_host_net_set(), test_extract_features_attack_host_pid_set(), test_extract_features_attack_latest_image_set() (+7 more)

### Community 16 - "Security Tools Scanner"
Cohesion: 0.23
Nodes (14): build_path_lookup(), _extract_repo_relpath(), log(), main(), scan_security_tools.py ======================= Run Checkov + kube-linter on all, Run kube-linter and return the number of failed checks., Extract (repo_name, relative_path) from a Rahman dataset path.     Handles two p, Build a map: rf_dataset.yaml_path → downloaded local_path.      The FINAL-COUNT (+6 more)

### Community 17 - "Graph Augmentation"
Cohesion: 0.26
Nodes (14): augment_edge_dropout(), augment_feature_mask(), generate_variants(), load_npz(), main(), augment_graphs.py ================== Graph augmentation to address the class imb, Sample a connected subgraph of approximately target_size nodes.     Starts BFS f, Generate n_variants augmented copies of a graph using a mix of strategies. (+6 more)

### Community 18 - "RF Classifier Inference"
Cohesion: 0.17
Nodes (11): ModelLoadError, ndarray, Path, _compute_derived_features(), _feats_to_rf_vec(), rf_classifier.py ================ Inference-only Random Forest wrapper.  Loads r, Compute cap_misuse, all_secrets, total_misconfigs from raw feature dict., Map yaml_parser output dict → 25-dim RF input vector. (+3 more)

### Community 19 - "RF Dataset Builder"
Cohesion: 0.26
Nodes (13): binarize(), build_dataset(), compute_risk_score(), extract_repo_name(), load_csv(), load_metrics_lookup(), main(), build_rf_dataset.py ==================== Build the Random Forest Layer 1 trainin (+5 more)

### Community 20 - "RF Dataset Enrichment"
Cohesion: 0.31
Nodes (12): badpods_category_from_path(), compute_risk_score(), compute_severity_class(), flags_to_row(), goat_scenario_from_path(), main(), process_badpods(), process_kubernetes_goat() (+4 more)

### Community 21 - "Cross-Val Metrics"
Cohesion: 0.26
Nodes (13): cv_metrics, accuracy_mean, accuracy_std, folds, macro_f1_mean, macro_f1_std, precision_mean, precision_std (+5 more)

### Community 22 - "Error Handling"
Cohesion: 0.24
Nodes (7): Exception, GraphBuildError, KubescanError, ManifestParseError, exceptions.py ============= Typed exception hierarchy for kubescan.  All kubesca, Base for all kubescan errors — catch this at the CLI boundary., Path

### Community 23 - "RF Training Results"
Cohesion: 0.18
Nodes (10): binary, loco_cv_metrics, oob_score, feature_names, severity, oob_score, target, binary_f1_achieved (+2 more)

### Community 24 - "Escape Signal Computation"
Cohesion: 0.24
Nodes (10): ndarray, compute_escape_fraction(), compute_escape_signal(), Fraction of nodes with at least one escape flag set.      Used for display and r, Binary signal: 1.0 if any node has at least one escape flag set, 0.0 otherwise., test_ga_ensemble.py =================== Unit tests for kubescan/model/ga_ensembl, test_escape_fraction_all_zeros_returns_zero(), test_escape_fraction_with_host_pid_flag_returns_one() (+2 more)

### Community 25 - "RF Model Training"
Cohesion: 0.35
Nodes (10): compute_metrics(), load_dataset(), main(), train_rf.py ============ Train Layer 1 Random Forest classifier on rf_dataset.cs, Leave-one-repo-out cross-validation.      Holds out each unique repo_name in tur, Load rf_dataset.csv and return (X, y_binary, y_severity, feature_names).     Ext, run_cv(), run_loco_cv() (+2 more)

### Community 26 - "HOSTPATH Data Fix"
Cohesion: 0.36
Nodes (9): _build_gitlab_lookup(), detect_hostpath(), _extract_repo_relpath(), _load_docs(), main(), patch_hostpath_column.py ========================= One-time script: adds HOSTPAT, Return 1 if the YAML file has any non-docker-sock hostPath volume., Build a mapping: rf_dataset yaml_path → local file path, for GitLab rows.      C (+1 more)

### Community 27 - "Attack Repo Ingestion"
Cohesion: 0.36
Nodes (8): compute_label(), compute_severity(), find_yamls(), main(), ingest_attack_repos.py ====================== Ingests newly cloned attack/securi, 0=clean, 1=misconfig (any flag set)., 0=clean, 1=low_medium, 2=high_critical., Path

### Community 28 - "RF Test Metrics"
Cohesion: 0.36
Nodes (9): test_metrics, test_metrics, accuracy, confusion_matrix, macro_f1, per_class_f1, precision, recall (+1 more)

### Community 29 - "RF Hyperparameters"
Cohesion: 0.22
Nodes (9): rf_params, class_weight, max_depth, max_features, min_samples_leaf, n_estimators, n_jobs, oob_score (+1 more)

### Community 30 - "Ensemble Checkpoint Loading"
Cohesion: 0.25
Nodes (5): Path, EnsembleScorer, Return ensemble score in [0, 1].          Parameters         ----------, Heuristic cluster label from probabilities.           2 (ATTACK_CHAIN)       if, Load GA-optimised weights and score a cluster given model predictions.      Para

### Community 31 - "Graph Builder Tests"
Cohesion: 0.39
Nodes (7): Path, test_graph_builder.py ===================== Unit tests for kubescan/utils/graph_, test_build_cluster_graph_attack_manifest_produces_escape_node(), test_build_cluster_graph_node_count_matches_input(), test_build_cluster_graph_node_data_length_matches_input(), test_graph_to_pyg_edge_index_has_two_rows(), test_graph_to_pyg_node_feature_shape()

### Community 32 - "Data Splits (5-Fold CV)"
Cohesion: 0.38
Nodes (6): k_fold_splits(), main(), create_splits.py ================= Create stratified train/val/test splits and 5, Stratified train/val/test split.     Maintains label distribution across splits., Stratified k-fold cross-validation.     Returns list of (train_names, val_names), stratified_split()

### Community 33 - "Test Fixtures"
Cohesion: 0.38
Nodes (6): Path, attack_yaml(), clean_yaml(), cluster_dir(), conftest.py =========== Shared pytest fixtures for kubescan tests., Directory containing both a clean and an attack manifest.

### Community 34 - "CLI Integration Tests"
Cohesion: 0.50
Nodes (3): test_cli_scan.py ================ Integration tests for the kubescan CLI scan co, kubescan scan --format json should exit 0 and emit valid JSON., test_cli_scan_produces_json_output()

### Community 36 - "CLI Entry Point"
Cohesion: 0.67
Nodes (3): Context, main(), Kubernetes attack-chain risk scanner using GNN + Random Forest ensemble.

## Knowledge Gaps
- **152 isolated node(s):** `Tensor`, `Path`, `device`, `ndarray`, `Data` (+147 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DataLoader` connect `GNN Dataset Pipeline` to `Inference Pipeline`, `GAT Model Training`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `run_gnn_ensemble()` connect `GAT Model Training` to `GNN Dataset Pipeline`, `kubescan CLI & Ensemble`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Why does `extract_cluster_features()` connect `Inference Pipeline` to `YAML Security Feature Extraction`?**
  _High betweenness centrality (0.071) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `KubeClusterDataset` (e.g. with `DataLoader` and `KubeClusterDataset`) actually correct?**
  _`KubeClusterDataset` has 8 INFERRED edges - model-reasoned connections that need verification._
- **What connects `kubescan — Kubernetes attack-chain risk scanner.  Loads trained GNN + Random For`, `cli.py ====== kubescan CLI — `kubescan scan <path>`  Entry point for the Kuberne`, `Resolve checkpoints directory: CLI arg → env var → package default.` to the rest of the system?**
  _314 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `GNN Dataset Pipeline` be split into smaller, more focused modules?**
  _Cohesion score 0.053555750658472345 - nodes in this community are weakly interconnected._
- **Should `YAML Security Feature Extraction` be split into smaller, more focused modules?**
  _Cohesion score 0.05288207297726071 - nodes in this community are weakly interconnected._