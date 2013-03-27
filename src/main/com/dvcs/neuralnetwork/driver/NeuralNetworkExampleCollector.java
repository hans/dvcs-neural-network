package com.dvcs.neuralnetwork.driver;

import com.dvcs.neuralnetwork.driver.DataQueueListener.NewDataCallback;

public class NeuralNetworkExampleCollector {

	static final String DATA_QUEUE_NAME = "robotData";

	private DataQueueListener queueListener;
	private NewDataCallback dataCallback;

	public NeuralNetworkExampleCollector(NewDataCallback dataCallback) {
		this.dataCallback = dataCallback;
	}

	public boolean isListening() {
		return queueListener != null && queueListener.isAlive();
	}

	public void startQueueListener() {
		queueListener = new DataQueueListener(DATA_QUEUE_NAME, dataCallback);
		queueListener.start();
	}

	public void stopQueueListener() {
		if ( queueListener == null )
			return;

		queueListener.stopListening();
	}

}
