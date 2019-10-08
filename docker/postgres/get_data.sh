!/bin/bash
wget -P /tmp/ https://archive.ics.uci.edu/ml/machine-learning-databases/00235/household_power_consumption.zip
unzip /tmp/household_power_consumption.zip
# rm -rf /tmp/*.zip
# sed '/^\s*$/d' /tmp/household_power_consumption.txt > /tmp/household_power_consumption.csv
# rm -rf /tmp/*.txt