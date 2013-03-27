package com.dvcs.neuralnetwork.driver;

import java.io.IOException;
import java.util.logging.Logger;

import com.rabbitmq.client.Channel;
import com.rabbitmq.client.Connection;
import com.rabbitmq.client.ConnectionFactory;
import com.rabbitmq.client.QueueingConsumer;
import com.rabbitmq.client.QueueingConsumer.Delivery;

public class DataQueueListener extends Thread {

	private static final String DEFAULT_HOST_NAME = "localhost";
	private static final long DELIVERY_TIMEOUT = 2000;
	
	private static final Logger LOGGER = Logger.getLogger("DataQueueListener");

	private String host;
	private String queueName;
	private NewDataCallback callback;
	
	private volatile boolean shouldStop = false;
	
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
			
			LOGGER.info("Queue listener up");
			
			while ( !shouldStop ) {
				Delivery delivery = consumer.nextDelivery(DELIVERY_TIMEOUT);
				
				if ( delivery != null )
					callback.receivedData(delivery.getBody());
			}
						
			LOGGER.info("Queue listener down");
			
			channel.close();
			connection.close();
		} catch ( InterruptedException e ) {
			e.printStackTrace();
		} catch ( IOException e ) {
			e.printStackTrace();
		}
	}
	
	public void stopListening() {
		shouldStop = true;
	}

}
