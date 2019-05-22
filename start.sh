# # test instance-based method
# python main.py gps_0.1_10.csv similar_values LOF 0.1 
# # show LOF graphs
# python LOF_graph.py ./results/LOF.txt 34 56 -69 -91

python main.py ./config/model_config.json

python LOF_graph.py ./results/result_data.csv_similar_values.json 35 55 -70 -90

python svm_graph.py ./results/result_data.csv_similar_values.json 4 86 -29 -151

# # test instance-based method
# python main.py gps_0.1.csv similar_values instance-based 0.1
# # show LOF graphs5
# python LOF_graph.py ./results/LOF.txt 43 47 -79 -83
