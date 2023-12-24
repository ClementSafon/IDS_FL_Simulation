data: data_csv_to_client.py
	python data_csv_to_client.py -n 3 -f $(name) -trf dataset/UNSW-NB15/a\ part\ of\ training\ and\ testing\ set/UNSW_NB15_training-set.csv -tef dataset/UNSW-NB15/a\ part\ of\ training\ and\ testing\ set/UNSW_NB15_testing-set.csv 

data_reverse: data_csv_to_client.py
	python data_csv_to_client.py -n 3 -f $(name) -tef dataset/UNSW-NB15/a\ part\ of\ training\ and\ testing\ set/UNSW_NB15_training-set.csv -trf dataset/UNSW-NB15/a\ part\ of\ training\ and\ testing\ set/UNSW_NB15_testing-set.csv 


run_ce: main_ce.py
	python main_ce.py -o $(name) -d $(data)

run_fe: main_de.py
	python main_de.py -o $(name) -d $(data)

show: show.py
	python show.py

eval: eval.py
	python eval.py --data_client_dir $(data) --final_path $(final)
	
clean:
	rm -fr data_client_*/
	rm -fr final_*/
