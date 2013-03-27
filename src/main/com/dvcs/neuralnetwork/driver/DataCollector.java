package com.dvcs.neuralnetwork.driver;

import com.dvcs.neuralnetwork.driver.DataQueueListener.NewDataCallback;

public class DataCollector {

	private String queueName;
	private DataQueueListener queueListener;
	private NewDataCallback dataCallback;

	public DataCollector(String queueName,
			NewDataCallback dataCallback) {
		this.queueName = queueName;
		this.dataCallback = dataCallback;
	}

	public boolean isListening() {
		return queueListener != null && queueListener.isAlive();
	}

	public void startQueueListener() {
		queueListener = new DataQueueListener(queueName, dataCallback);
		queueListener.start();
	}

	public void stopQueueListener() {
		if ( queueListener == null )
			return;

		queueListener.stopListening();
	}

}
