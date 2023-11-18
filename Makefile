data: data_preprocessing.py
	rm -f 'data_party*' 
	python data_preprocessing.py -n 3 -trf dataset/UNSW-NB15/a\ part\ of\ training\ and\ testing\ set/UNSW_NB15_training-set.csv -tef dataset/UNSW-NB15/a\ part\ of\ training\ and\ testing\ set/UNSW_NB15_testing-set.csv 

model: model.py
	rm -f 'template_fl_model.keras' 
	python model.py

run: main.py
	rm -f 'final_fl*centralized*'
	python main.py

run_fe: main_fe.py
	rm -f fl_model.keras
	rm -f history.json
	python main_fe.py

show: show.py
	python show.py

eval: eval.py
	python eval.py