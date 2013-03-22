package com.dvcs.neuralnetwork.driver;

import java.io.IOException;

import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
import com.rabbitmq.client.QueueingConsumer;

public class DataQueueListener implements Runnable {

	private static final String DEFAULT_HOST_NAME = "localhost";

	private String host;
	private String queueName;
	private NewDataCallback callback;
	
	public interface NewDataCallback {
		public void receivedData(byte[] data);
	}

	public DataQueueListener(String queueName, NewDataCallback callback) {
		this(DEFAULT_HOST_NAME, queueName, callback);
	}

	public DataQueueListener(String host, String queueName,
			NewDataCallback callback) {
		this.host = host;
		this.queueName = queueName;
		this.callback = callback;
	}

	public void run() {
		ConnectionFactory factory = new ConnectionFactory();
		factory.setHost(host);

		try {
			Connection connection = factory.newConnection();
			Channel channel = connection.createChannel();
			channel.queueDeclare(queueName, false, false, false, null);
			
			QueueingConsumer consumer = new QueueingConsumer(channel);
			channel.basicConsume(queueName, true, consumer);
			
			while ( true ) {
				QueueingConsumer.Delivery delivery = consumer.nextDelivery();
				callback.receivedData(delivery.getBody());
			}
		} catch ( InterruptedException e ) {
			e.printStackTrace();
		} catch ( IOException e ) {
			e.printStackTrace();
		}
	}

}
