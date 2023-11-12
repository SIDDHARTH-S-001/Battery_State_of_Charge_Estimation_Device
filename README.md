# Battery_State_of_Charge_Estimation_Device

<h3>Introduction</h3>
<br>This is a deep learning approach to estimate State-of-Charge of 18650 Li-Ion batteries in real-time with high accuracy.</br>

<br>The dataset we used can be found here, [Dataset](https://data.mendeley.com/datasets/cp3473x7xv/1).<br>
<br>Clone this repo into your working directory and execute the **training_code.m** file to train an artifical neural network.</br>
<br>You can change the network hyper-parameters to obtain different training results.</br>
<br>Once the training is complete, you can export the model into various formats as per your use case through builtin matlab commands. In this case, it has been exported as a Tensorflow model and later converted to TFLite format to be deployed on Hardware.</br>
<br>Matlab Code files have been written in Matlab 2020b and all python files have been verified to work in Python 3.9</br>

<br>The Schematic and board files for the PCB HAT designed for Raspberry Pi can be found in the [PCB](https://github.com/SIDDHARTH-S-001/Battery_State_of_Charge_Estimation_Device/tree/main/PCB) folder. These were designed in [Eagle](https://www.autodesk.com/products/eagle/overview?term=1-YEAR&tab=subscription) 9.6.2</br>

<h3>Components</h3>
<br>1) Raspberry Pi 4</br>
<br>2) 3Ah 18650 Li-Ion cell</br>
<br>3) 0-25V Generic Voltage sensor</br>
<br>4) 0-30A ACS712 Current Sensor</br>
<br>5) DHT11 Temperature Humidity Sensor</br>
<br>6) ADS1115 / MCP3208 (ADC)</br>


