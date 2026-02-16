// Neo4j Cypher Import Script for Xiaohao Cai Paper Knowledge Graph
// Version: 1.0
// Run with: cypher-shell -u neo4j -p password -f neo4j_import.cypher

// Clear existing data
MATCH (n) DETACH DELETE n;

// Create uniqueness constraints
CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE;
CREATE CONSTRAINT author_id_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE;
CREATE CONSTRAINT method_id_unique IF NOT EXISTS FOR (m:Method) REQUIRE m.id IS UNIQUE;
CREATE CONSTRAINT domain_id_unique IF NOT EXISTS FOR (d:Domain) REQUIRE d.id IS UNIQUE;
CREATE CONSTRAINT tool_id_unique IF NOT EXISTS FOR (t:Tool) REQUIRE t.id IS UNIQUE;

// ============================================
// CREATE AUTHORS
// ============================================
CREATE (a001:Author {id: 'A001', name: 'Xiaohao Cai', affiliation: 'University of Southampton', role: 'primary'});
CREATE (a002:Author {id: 'A002', name: 'Raymond H. F. Chan', affiliation: 'City University of Hong Kong', role: 'collaborator'});
CREATE (a003:Author {id: 'A003', name: 'Tieyong Zeng', affiliation: 'Chinese University of Hong Kong', role: 'collaborator'});
CREATE (a004:Author {id: 'A004', name: 'Carola-Bibiane Schonlieb', affiliation: 'University of Cambridge', role: 'collaborator'});
CREATE (a005:Author {id: 'A005', name: 'Gabriele Steidl', affiliation: 'University of Kaiserslautern', role: 'collaborator'});
CREATE (a006:Author {id: 'A006', name: 'Jason D. McEwen', affiliation: 'University College London', role: 'collaborator'});
CREATE (a007:Author {id: 'A007', name: 'Marcelo Pereyra', affiliation: 'Heriot-Watt University', role: 'collaborator'});
CREATE (a008:Author {id: 'A008', name: 'Mila Nikolova', affiliation: 'ENS Paris', role: 'collaborator'});

// ============================================
// CREATE METHODS
// ============================================
CREATE (m001:Method {id: 'M001', name: 'ROF', full_name: 'Rudin-Osher-Fatemi Model', category: 'Variational', description: 'Total Variation denoising model'});
CREATE (m002:Method {id: 'M002', name: 'Mumford-Shah', full_name: 'Mumford-Shah Functional', category: 'Variational', description: 'Variational segmentation model'});
CREATE (m003:Method {id: 'M003', name: 'SLaT', full_name: 'Smoothing Lifting Thresholding', category: 'Segmentation', description: 'Three-stage segmentation framework'});
CREATE (m004:Method {id: 'M004', name: 'T-ROF', full_name: 'Thresholded ROF', category: 'Segmentation', description: 'ROF with thresholding for segmentation'});
CREATE (m005:Method {id: 'M005', name: 'SaT', full_name: 'Smoothing and Thresholding', category: 'Segmentation', description: 'Two-stage segmentation paradigm'});
CREATE (m006:Method {id: 'M006', name: 'Tight-Frame', full_name: 'Tight Frame Wavelet', category: 'Wavelet', description: 'Wavelet-based representation'});
CREATE (m007:Method {id: 'M007', name: 'Graph-Cut', full_name: 'Graph Cut Optimization', category: 'Optimization', description: 'Combinatorial optimization'});
CREATE (m008:Method {id: 'M008', name: 'MCGC', full_name: 'Multi-Class Graph Cut', category: 'Optimization', description: 'Multi-class graph cut'});
CREATE (m009:Method {id: 'M009', name: 'Split-Bregman', full_name: 'Split Bregman Algorithm', category: 'Optimization', description: 'Iterative TV optimization'});
CREATE (m010:Method {id: 'M010', name: 'Primal-Dual', full_name: 'Primal-Dual Algorithm', category: 'Optimization', description: 'Chambolle-Pock algorithm'});
CREATE (m011:Method {id: 'M011', name: 'Tucker', full_name: 'Tucker Decomposition', category: 'Tensor', description: 'Higher-order SVD'});
CREATE (m012:Method {id: 'M012', name: 'Tensor-Train', full_name: 'Tensor Train Decomposition', category: 'Tensor', description: 'Matrix product state'});
CREATE (m013:Method {id: 'M013', name: 'CUR', full_name: 'CUR Decomposition', category: 'Tensor', description: 'Interpolative decomposition'});
CREATE (m014:Method {id: 'M014', name: 'Sketching', full_name: 'Random Sketching', category: 'Approximation', description: 'Randomized low-rank approximation'});
CREATE (m015:Method {id: 'M015', name: 'Proximal-MCMC', full_name: 'Proximal MCMC', category: 'Bayesian', description: 'MCMC with proximal operators'});
CREATE (m016:Method {id: 'M016', name: 'Varifold', full_name: 'Neural Varifolds', category: 'Deep Learning', description: 'Point cloud geometry'});
CREATE (m017:Method {id: 'M017', name: 'Mamba', full_name: 'Mamba Architecture', category: 'Deep Learning', description: 'State space model'});
CREATE (m018:Method {id: 'M018', name: 'Diffusion-Model', full_name: 'Diffusion Models', category: 'Deep Learning', description: 'Generative diffusion'});
CREATE (m019:Method {id: 'M019', name: 'Few-Shot', full_name: 'Few-Shot Learning', category: 'Deep Learning', description: 'Learning from limited data'});
CREATE (m020:Method {id: 'M020', name: 'LoRA', full_name: 'Low-Rank Adaptation', category: 'Deep Learning', description: 'Parameter-efficient fine-tuning'});
CREATE (m021:Method {id: 'M021', name: 'HOOI', full_name: 'Higher-Order Orthogonal Iteration', category: 'Tensor', description: 'Tucker decomposition algorithm'});
CREATE (m022:Method {id: 'M022', name: 'K-means', full_name: 'K-Means Clustering', category: 'Clustering', description: 'Vector quantization'});
CREATE (m023:Method {id: 'M023', name: 'Convex-Relaxation', full_name: 'Convex Relaxation', category: 'Optimization', description: 'Discrete to continuous'});
CREATE (m024:Method {id: 'M024', name: 'Lab-Color', full_name: 'Lab Color Space', category: 'Preprocessing', description: 'Perceptually uniform color'});

// ============================================
// CREATE DOMAINS
// ============================================
CREATE (d001:Domain {id: 'D001', name: 'Medical Imaging', description: 'Medical image analysis'});
CREATE (d002:Domain {id: 'D002', name: 'Remote Sensing', description: 'Satellite image analysis'});
CREATE (d003:Domain {id: 'D003', name: 'Radio Astronomy', description: 'Radio interferometric imaging'});
CREATE (d004:Domain {id: 'D004', name: 'Computer Vision', description: 'General CV tasks'});
CREATE (d005:Domain {id: 'D005', name: '3D Vision', description: 'Point cloud processing'});
CREATE (d006:Domain {id: 'D006', name: 'NLP', description: 'Natural language processing'});
CREATE (d007:Domain {id: 'D007', name: 'HAR', description: 'Human activity recognition'});
CREATE (d008:Domain {id: 'D008', name: 'Personality Detection', description: 'Personality prediction'});
CREATE (d009:Domain {id: 'D009', name: 'Fake News Detection', description: 'Misinformation detection'});
CREATE (d010:Domain {id: 'D010', name: 'Motion Generation', description: '3D motion synthesis'});
CREATE (d011:Domain {id: 'D011', name: 'Object Detection', description: '2D/3D object detection'});
CREATE (d012:Domain {id: 'D012', name: 'Multi-Object Tracking', description: 'Video object tracking'});

// ============================================
// CREATE TOOLS
// ============================================
CREATE (t001:Tool {id: 'T001', name: 'Python', category: 'Programming Language'});
CREATE (t002:Tool {id: 'T002', name: 'MATLAB', category: 'Programming Language'});
CREATE (t003:Tool {id: 'T003', name: 'PyTorch', category: 'Deep Learning Framework'});
CREATE (t004:Tool {id: 'T004', name: 'TensorFlow', category: 'Deep Learning Framework'});
CREATE (t005:Tool {id: 'T005', name: 'NumPy', category: 'Numerical Library'});
CREATE (t006:Tool {id: 'T006', name: 'SciPy', category: 'Scientific Library'});
CREATE (t007:Tool {id: 'T007', name: 'OpenCV', category: 'Computer Vision Library'});
CREATE (t008:Tool {id: 'T008', name: 'TensorLy', category: 'Tensor Library'});
CREATE (t009:Tool {id: 'T009', name: 'scikit-learn', category: 'ML Library'});
CREATE (t010:Tool {id: 'T010', name: 'CUDA', category: 'GPU Computing'});

// ============================================
// CREATE PAPERS (Core Papers)
// ============================================
CREATE (p001:Paper {id: 'P001', title: 'Tight-Frame Vessel Segmentation', year: 2011, arxiv: '1109.0217', venue: 'SIAM J. Imaging Sci.'});
CREATE (p002:Paper {id: 'P002', title: 'Two-Stage Segmentation', year: 2013, venue: 'SIAM J. Imaging Sci.'});
CREATE (p003:Paper {id: 'P003', title: 'Variational Segmentation-Restoration', year: 2014, arxiv: '1405.2128', venue: 'Pattern Recognition'});
CREATE (p004:Paper {id: 'P004', title: 'LiDAR-Hyperspectral Registration', year: 2014, arxiv: '1410.0226', venue: 'IEEE TGRS'});
CREATE (p005:Paper {id: 'P005', title: 'SLaT Three-stage Segmentation', year: 2015, arxiv: '1506.00060', venue: 'IEEE TIP'});
CREATE (p006:Paper {id: 'P006', title: 'Wavelet Segmentation on Sphere', year: 2016, arxiv: '1609.06500', venue: 'SIAM J. Imaging Sci.'});
CREATE (p007:Paper {id: 'P007', title: '3D Tree Delineation Graph Cut', year: 2017, arxiv: '1701.06715', venue: 'Remote Sens. Environ.'});
CREATE (p008:Paper {id: 'P008', title: 'Radio Interferometric Imaging I', year: 2017, arxiv: '1711.04818', venue: 'MNRAS'});
CREATE (p009:Paper {id: 'P009', title: 'Radio Interferometric Imaging II', year: 2017, arxiv: '1711.04819', venue: 'MNRAS'});
CREATE (p010:Paper {id: 'P010', title: 'Online Radio Interferometric Imaging', year: 2017, arxiv: '1712.04462', venue: 'MNRAS'});
CREATE (p011:Paper {id: 'P011', title: 'Mumford-Shah and ROF Linkage', year: 2018, arxiv: '1807.10194', venue: 'SIAM J. Imaging Sci.'});
CREATE (p012:Paper {id: 'P012', title: 'High-Dimensional Inverse Problems', year: 2018, arxiv: '1811.02514', venue: 'arXiv'});
CREATE (p013:Paper {id: 'P013', title: '3D Tree Segmentation MCGC', year: 2019, arxiv: '1903.08481', venue: 'IEEE TGRS'});
CREATE (p014:Paper {id: 'P014', title: 'Two-stage High-dimensional Classification', year: 2019, arxiv: '1905.08538', venue: 'M3AS'});
CREATE (p015:Paper {id: 'P015', title: '3D Orientation Field Transform', year: 2020, arxiv: '2010.01453', venue: 'Pattern Anal. Appl.'});
CREATE (p016:Paper {id: 'P016', title: 'Proximal Nested Sampling', year: 2021, arxiv: '2106.03646', venue: 'arXiv'});
CREATE (p017:Paper {id: 'P017', title: 'Practical Sketching Tucker Approximation', year: 2023, arxiv: '2301.11598', venue: 'SIAM J. Sci. Comput.'});
CREATE (p018:Paper {id: 'P018', title: 'GO-LDA Generalised Optimal LDA', year: 2023, arxiv: '2305.14568', venue: 'arXiv'});
CREATE (p019:Paper {id: 'P019', title: 'Semantic Segmentation by Proportions', year: 2023, arxiv: '2305.15608', venue: 'arXiv'});
CREATE (p020:Paper {id: 'P020', title: 'Few-shot Medical Imaging Inference', year: 2023, arxiv: '2306.11152', venue: 'arXiv'});
CREATE (p021:Paper {id: 'P021', title: 'Bilevel Peer-Reviewing Problem', year: 2023, arxiv: '2307.12248', venue: 'arXiv'});
CREATE (p022:Paper {id: 'P022', title: 'Tensor Train Approximation', year: 2023, arxiv: '2308.01480', venue: 'arXiv'});
CREATE (p023:Paper {id: 'P023', title: 'IIHT Medical Report Generation', year: 2023, arxiv: '2308.05633', venue: 'arXiv'});
CREATE (p024:Paper {id: 'P024', title: 'TransNet Transfer Learning HAR', year: 2023, arxiv: '2309.06951', venue: 'arXiv'});
CREATE (p025:Paper {id: 'P025', title: 'Equalizing Protected Attributes', year: 2023, arxiv: '2311.14733', venue: 'arXiv'});
CREATE (p026:Paper {id: 'P026', title: 'Non-negative Subspace Few-Shot', year: 2024, arxiv: '2404.02656', venue: 'arXiv'});
CREATE (p027:Paper {id: 'P027', title: 'Discrepancy-based Diffusion Brain MRI', year: 2024, arxiv: '2405.04974', venue: 'arXiv'});
CREATE (p028:Paper {id: 'P028', title: 'Detect Closer Surfaces 3D', year: 2024, arxiv: '2407.04061', venue: 'arXiv'});
CREATE (p029:Paper {id: 'P029', title: 'CNNs RNNs Transformers HAR Survey', year: 2024, arxiv: '2407.06162', venue: 'arXiv'});
CREATE (p030:Paper {id: 'P030', title: 'Cross-Domain LiDAR Detection', year: 2024, arxiv: '2408.12708', venue: 'arXiv'});
CREATE (p031:Paper {id: 'P031', title: 'Talk2Radar Language-Radar Multimodal', year: 2025, arxiv: '2405.12821', venue: 'arXiv'});
CREATE (p032:Paper {id: 'P032', title: 'Neural Varifolds Point Cloud', year: 2025, arxiv: '2407.04844', venue: 'arXiv'});
CREATE (p033:Paper {id: 'P033', title: 'GAMED Multimodal Fake News', year: 2025, arxiv: '2412.12164', venue: 'arXiv'});
CREATE (p034:Paper {id: 'P034', title: 'tCURLoRA Tensor CUR Medical', year: 2025, arxiv: '2501.02227', venue: 'arXiv'});
CREATE (p035:Paper {id: 'P035', title: 'Concept-Based XAI Metrics', year: 2025, arxiv: '2501.19271', venue: 'arXiv'});
CREATE (p036:Paper {id: 'P036', title: 'LL4G Graph-Based Personality', year: 2025, arxiv: '2504.02146', venue: 'arXiv'});
CREATE (p037:Paper {id: 'P037', title: 'CornerPoint3D Nearest Corner', year: 2025, arxiv: '2504.02464', venue: 'arXiv'});
CREATE (p038:Paper {id: 'P038', title: 'Less but Better PEFT Personality', year: 2025, arxiv: '2504.05411', venue: 'arXiv'});
CREATE (p039:Paper {id: 'P039', title: 'MOGO 3D Motion Generation', year: 2025, arxiv: '2506.05952', venue: 'arXiv'});
CREATE (p040:Paper {id: 'P040', title: 'GRASPTrack Multi-Object Tracking', year: 2025, arxiv: '2508.08117', venue: 'arXiv'});
CREATE (p041:Paper {id: 'P041', title: 'HiFi-Mamba MRI Reconstruction', year: 2025, arxiv: '2508.09179', venue: 'arXiv'});
CREATE (p042:Paper {id: 'P042', title: 'EmoPerso Emotion-Aware Personality', year: 2025, arxiv: '2509.02450', venue: 'arXiv'});
CREATE (p043:Paper {id: 'P043', title: 'HIPPD Brain-Inspired Personality', year: 2025, arxiv: '2510.09893', venue: 'arXiv'});
CREATE (p044:Paper {id: 'P044', title: '3D Growth Trajectory Reconstruction', year: 2025, arxiv: '2511.02142', venue: 'arXiv'});
CREATE (p045:Paper {id: 'P045', title: 'MotionDuet 3D Motion Generation', year: 2025, arxiv: '2511.18209', venue: 'arXiv'});
CREATE (p046:Paper {id: 'P046', title: 'HiFi-MambaV2 Hierarchical MRI', year: 2025, arxiv: '2511.18534', venue: 'arXiv'});
CREATE (p047:Paper {id: 'P047', title: 'CALM Culturally Self-Aware LLMs', year: 2026, arxiv: '2601.03483', venue: 'arXiv'});

// ============================================
// CREATE RELATIONSHIPS: Paper USES Method
// ============================================
MATCH (p:Paper {id: 'P001'}), (m:Method {id: 'M006'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P002'}), (m:Method {id: 'M005'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P002'}), (m:Method {id: 'M001'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P003'}), (m:Method {id: 'M002'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P003'}), (m:Method {id: 'M001'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P005'}), (m:Method {id: 'M003'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P005'}), (m:Method {id: 'M001'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P005'}), (m:Method {id: 'M002'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P005'}), (m:Method {id: 'M022'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P005'}), (m:Method {id: 'M024'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P007'}), (m:Method {id: 'M007'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P007'}), (m:Method {id: 'M008'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P008'}), (m:Method {id: 'M015'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P008'}), (m:Method {id: 'M001'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P009'}), (m:Method {id: 'M015'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P010'}), (m:Method {id: 'M015'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P011'}), (m:Method {id: 'M002'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P011'}), (m:Method {id: 'M001'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P011'}), (m:Method {id: 'M004'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P013'}), (m:Method {id: 'M008'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P015'}), (m:Method {id: 'M006'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P016'}), (m:Method {id: 'M015'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P017'}), (m:Method {id: 'M011'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P017'}), (m:Method {id: 'M014'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P017'}), (m:Method {id: 'M021'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P019'}), (m:Method {id: 'M003'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P020'}), (m:Method {id: 'M019'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P022'}), (m:Method {id: 'M012'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P022'}), (m:Method {id: 'M014'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P027'}), (m:Method {id: 'M018'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P030'}), (m:Method {id: 'M016'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P032'}), (m:Method {id: 'M016'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P034'}), (m:Method {id: 'M013'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P034'}), (m:Method {id: 'M020'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P041'}), (m:Method {id: 'M017'}) CREATE (p)-[:USES]->(m);
MATCH (p:Paper {id: 'P046'}), (m:Method {id: 'M017'}) CREATE (p)-[:USES]->(m);

// ============================================
// CREATE RELATIONSHIPS: Paper APPLIES_TO Domain
// ============================================
MATCH (p:Paper {id: 'P001'}), (d:Domain {id: 'D001'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P004'}), (d:Domain {id: 'D002'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P007'}), (d:Domain {id: 'D002'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P008'}), (d:Domain {id: 'D003'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P009'}), (d:Domain {id: 'D003'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P010'}), (d:Domain {id: 'D003'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P013'}), (d:Domain {id: 'D002'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P015'}), (d:Domain {id: 'D005'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P020'}), (d:Domain {id: 'D001'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P023'}), (d:Domain {id: 'D001'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P024'}), (d:Domain {id: 'D007'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P027'}), (d:Domain {id: 'D001'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P028'}), (d:Domain {id: 'D005'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P029'}), (d:Domain {id: 'D007'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P030'}), (d:Domain {id: 'D005'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P030'}), (d:Domain {id: 'D011'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P031'}), (d:Domain {id: 'D004'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P033'}), (d:Domain {id: 'D009'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P036'}), (d:Domain {id: 'D008'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P037'}), (d:Domain {id: 'D005'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P037'}), (d:Domain {id: 'D011'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P038'}), (d:Domain {id: 'D008'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P039'}), (d:Domain {id: 'D010'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P040'}), (d:Domain {id: 'D012'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P041'}), (d:Domain {id: 'D001'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P042'}), (d:Domain {id: 'D008'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P043'}), (d:Domain {id: 'D008'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P044'}), (d:Domain {id: 'D005'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P045'}), (d:Domain {id: 'D010'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P046'}), (d:Domain {id: 'D001'}) CREATE (p)-[:APPLIES_TO]->(d);
MATCH (p:Paper {id: 'P047'}), (d:Domain {id: 'D006'}) CREATE (p)-[:APPLIES_TO]->(d);

// ============================================
// CREATE RELATIONSHIPS: Author WRITES Paper
// ============================================
MATCH (a:Author {id: 'A001'}), (p:Paper) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A002'}), (p:Paper {id: 'P005'}) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A003'}), (p:Paper {id: 'P011'}) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A004'}), (p:Paper {id: 'P008'}) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A004'}), (p:Paper {id: 'P009'}) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A004'}), (p:Paper {id: 'P010'}) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A005'}), (p:Paper {id: 'P002'}) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A006'}), (p:Paper {id: 'P008'}) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A006'}), (p:Paper {id: 'P009'}) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A006'}), (p:Paper {id: 'P010'}) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A007'}), (p:Paper {id: 'P008'}) CREATE (a)-[:WRITES]->(p);
MATCH (a:Author {id: 'A008'}), (p:Paper {id: 'P005'}) CREATE (a)-[:WRITES]->(p);

// ============================================
// CREATE RELATIONSHIPS: Method INHERITS_FROM Method
// ============================================
MATCH (m1:Method {id: 'M003'}), (m2:Method {id: 'M005'}) CREATE (m1)-[:INHERITS_FROM]->(m2);
MATCH (m1:Method {id: 'M004'}), (m2:Method {id: 'M001'}) CREATE (m1)-[:INHERITS_FROM]->(m2);
MATCH (m1:Method {id: 'M008'}), (m2:Method {id: 'M007'}) CREATE (m1)-[:INHERITS_FROM]->(m2);
MATCH (m1:Method {id: 'M013'}), (m2:Method {id: 'M011'}) CREATE (m1)-[:INHERITS_FROM]->(m2);

// ============================================
// CREATE RELATIONSHIPS: Paper CITES Paper
// ============================================
MATCH (p1:Paper {id: 'P005'}), (p2:Paper {id: 'P002'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P005'}), (p2:Paper {id: 'P003'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P011'}), (p2:Paper {id: 'P002'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P011'}), (p2:Paper {id: 'P005'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P013'}), (p2:Paper {id: 'P007'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P017'}), (p2:Paper {id: 'P011'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P019'}), (p2:Paper {id: 'P005'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P022'}), (p2:Paper {id: 'P017'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P030'}), (p2:Paper {id: 'P028'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P034'}), (p2:Paper {id: 'P017'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P034'}), (p2:Paper {id: 'P022'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P041'}), (p2:Paper {id: 'P027'}) CREATE (p1)-[:CITES]->(p2);
MATCH (p1:Paper {id: 'P046'}), (p2:Paper {id: 'P041'}) CREATE (p1)-[:CITES]->(p2);

// ============================================
// CREATE RELATIONSHIPS: Paper USES_TOOL Tool
// ============================================
MATCH (p:Paper {id: 'P005'}), (t:Tool {id: 'T002'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P008'}), (t:Tool {id: 'T002'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P017'}), (t:Tool {id: 'T001'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P017'}), (t:Tool {id: 'T005'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P020'}), (t:Tool {id: 'T003'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P027'}), (t:Tool {id: 'T003'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P030'}), (t:Tool {id: 'T003'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P032'}), (t:Tool {id: 'T003'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P034'}), (t:Tool {id: 'T003'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P037'}), (t:Tool {id: 'T003'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P039'}), (t:Tool {id: 'T003'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P041'}), (t:Tool {id: 'T003'}) CREATE (p)-[:USES_TOOL]->(t);
MATCH (p:Paper {id: 'P046'}), (t:Tool {id: 'T003'}) CREATE (p)-[:USES_TOOL]->(t);

// ============================================
// VERIFY CREATION
// ============================================
RETURN 
    (SELECT count(*) FROM nodes) as total_nodes,
    (SELECT count(*) FROM relationships) as total_relationships;
