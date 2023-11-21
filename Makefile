data: data_preprocessing.py
	python data_preprocessing.py -n 3 -f $(folder) -trf dataset/UNSW-NB15/a\ part\ of\ training\ and\ testing\ set/UNSW_NB15_training-set.csv -tef dataset/UNSW-NB15/a\ part\ of\ training\ and\ testing\ set/UNSW_NB15_testing-set.csv 

run_ce: main_ce.py
	python main_ce.py -o $(folder) -d $(data)

run_fe: main_fe.py
	python main_fe.py -o $(folder) -d $(data)

show: show.py
	python show.py

eval: eval.py
	python eval.py --dir_client $(data) --dir_path $(final)
	
clean:
	rm -fr data_client_*/
	rm -fr final_*/
