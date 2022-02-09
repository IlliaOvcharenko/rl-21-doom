
echo "Scenario - basic, decay 0.9"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="basic" --encoder="usual" --epsilon_decay=0.9
echo "Scenario - basic, decay 0.99"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="basic" --encoder="usual" --epsilon_decay=0.99
echo "Scenario - basic, decay 0.999"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="basic" --encoder="usual" --epsilon_decay=0.999

echo "Scenario - deadly_corridor, decay 0.9"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="deadly_corridor" --encoder="usual" --epsilon_decay=0.9
echo "Scenario - deadly_corridor, decay 0.99"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="deadly_corridor" --encoder="usual" --epsilon_decay=0.99
echo "Scenario - deadly_corridor, decay 0.999"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="deadly_corridor" --encoder="usual" --epsilon_decay=0.999


echo "Scenario - basic, replay size 256"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="basic" --encoder="usual" --epsilon_decay=0.994 --replay_size=256
echo "Scenario - basic, replay size 1024"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="basic" --encoder="usual" --epsilon_decay=0.994 --replay_size=1024
echo "Scenario - basic, replay size 4096"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="basic" --encoder="usual" --epsilon_decay=0.994 --replay_size=4096

echo "Scenario - deadly_corridor, replay size 256"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="deadly_corridor" --encoder="usual" --epsilon_decay=0.994 --replay_size=256
echo "Scenario - deadly_corridor, replay size 1024"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="deadly_corridor" --encoder="usual" --epsilon_decay=0.994 --replay_size=1024
echo "Scenario - deadly_corridor, replay size 4096"
python scripts/run_train.py --n_epoches=15 --n_episodes_to_play=5 --game_scenario="deadly_corridor" --encoder="usual" --epsilon_decay=0.994 --replay_size=4096

