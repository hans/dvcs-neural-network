package com.dvcs.neuralnetwork.driver;

import java.util.logging.Logger;

import org.zeromq.ZMQ;
import org.zeromq.ZMQ.Context;
import org.zeromq.ZMQ.Socket;

public class DataQueueListener extends Thread {

	private static final String ZMQ_ADDRESS = "ipc:///tmp/robotData";
	private static final int ZMQ_THREADS = 1;

	private static final Logger LOGGER = Logger.getLogger("DataQueueListener");

	private String address;
	private NewDataCallback callback;

	private volatile boolean shouldStop = false;

	public interface NewDataCallback {
		public void receivedData(byte[] data);
	}

	public DataQueueListener(NewDataCallback callback) {
		this(ZMQ_ADDRESS, callback);
	}

	public DataQueueListener(String address, NewDataCallback callback) {
		this.address = address;
		this.callback = callback;
	}

	public void run() {
		Context context = ZMQ.context(ZMQ_THREADS);

		Socket client = context.socket(ZMQ.SUB);
		client.bind(address);

		LOGGER.info("Queue listener up");
		
		while ( !shouldStop ) {
			byte[] data = client.recv(ZMQ.DONTWAIT);
			
			if ( data != null )
				callback.receivedData(data);
			
			try {
				sleep(100);
			} catch ( InterruptedException e ) {
				e.printStackTrace();
			}
		}

		client.disconnect(address);
		client.close();
		
		LOGGER.info("Queue listener down");
	}

	public void stopListening() {
		shouldStop = true;
	}

}
