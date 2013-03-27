package com.dvcs.neuralnetwork.driver;

import com.dvcs.neuralnetwork.driver.DataQueueListener.NewDataCallback;

public class NeuralNetworkExampleCollector {

	static final String DATA_QUEUE_NAME = "robotData";
	
	private Thread dataListenerThread;
	private NewDataCallback dataCallback;
	
	public NeuralNetworkExampleCollector(NewDataCallback dataCallback) {
		this.dataCallback = dataCallback;
	}
	
	public void startQueueListener() {
		DataQueueListener listener = new DataQueueListener(DATA_QUEUE_NAME,
				dataCallback);

		dataListenerThread = new Thread(listener);
		dataListenerThread.start();
	}
	
}
