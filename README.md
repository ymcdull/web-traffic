My main solution for this "web traffic prediction" challenge is weighted average with three mothods: fibonacci median, holidays, and RNN.

These three methods can be got from:
fibo_submit.py => average of median solution, the number of days is generated from fibonacci sequence
holiday_submit.py => follows the assumption that the data distribution in holidays are much different with normal days
lstm_train.py & lstm_predict.py => RNN is based on 2 level LSTM structure given last 120 days history data.


Other Jupyter notebooks are some data preprocessing, and basic explorations.

Final submission is weighted average with fibo, holiday and 3 RNN results with "sgd" & "adam" optimizer.
