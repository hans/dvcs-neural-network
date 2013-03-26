package com.dvcs.neuralnetwork.driver;

import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;

import org.jblas.DoubleMatrix;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.MinimizerListener;

public class NeuralNetworkDriverGUI {

	static final int NUM_CLASSES = 10;
	static final String[] classLabels = new String[] { "1", "2", "3", "4", "5",
			"6", "7", "8", "9", "0" };

	JFrame frame;
	MatrixImageApplet applet;

	JPanel sidebar;
	JProgressBar[] classBars;
	JLabel predictionLabel;

	JPanel trainingSidebar;
	JLabel costLabel;

	NeuralNetworkDriver driver = new NeuralNetworkDriver(
			new DataQueueListener.NewDataCallback() {
				public void receivedData(byte[] data) {
					BufferedImage im = null;
					try {
						im = ImageIO.read(new ByteArrayInputStream(data));
					} catch ( IOException e ) {
						e.printStackTrace();
						return;
					}

					DoubleMatrix next = ImageConverter.convertImageToMatrix(im,
							true);
					next = ImageConverter.normalize(next);
				}
			});

	public void init() {
		frame = new JFrame();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(1010, 400);
		frame.getContentPane().setLayout(
				new BoxLayout(frame.getContentPane(), BoxLayout.X_AXIS));

		applet = new MatrixImageApplet();
		applet.init();
		applet.start();
		frame.add(applet);

		JPanel adminBar = initAdminBar();
		sidebar = initSidebar();
		trainingSidebar = initTrainingSidebar();

		frame.add(adminBar);
		frame.add(new JScrollPane(sidebar));
		frame.add(trainingSidebar);
		frame.setVisible(true);
	}

	private JPanel initAdminBar() {
		JPanel adminBar = new JPanel();
		GridLayout layout = new GridLayout(0, 2);
		adminBar.setLayout(layout);

		JButton startListeningButton = new JButton("Start data listener");
		startListeningButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				driver.startQueueListener();
			}
		});
		adminBar.add(startListeningButton);

		return adminBar;
	}

	private JPanel initSidebar() {
		JPanel sidebar = new JPanel();

		GridLayout layout = new GridLayout(0, 2);
		sidebar.setLayout(layout);

		JButton placeholderButton = new JButton("");
		sidebar.add(placeholderButton);

		predictionLabel = new JLabel();
		predictionLabel.setFont(new Font("Sans Serif", Font.PLAIN, 18));
		predictionLabel.setHorizontalAlignment(SwingConstants.CENTER);
		sidebar.add(predictionLabel);

		// hack
		sidebar.add(new JLabel(""));

		classBars = new JProgressBar[NUM_CLASSES];
		for ( int i = 0; i < NUM_CLASSES; i++ ) {
			classBars[i] = new JProgressBar(0, 1000);

			sidebar.add(new JLabel(classLabels[i]));
			sidebar.add(classBars[i]);
		}

		return sidebar;
	}

	private JPanel initTrainingSidebar() {
		JPanel sidebar = new JPanel();
		sidebar.setLayout(new GridLayout(0, 1));

		final MinimizerListener listener = new MinimizerListener() {
			public void minimizationIterationFinished(final int n,
					final double cost, DoubleVector parameters) {
				SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						costLabel.setText("Iteration " + n + "\n" + "Cost: "
								+ String.format("%.04f", cost));
					}
				});
			}
		};

		final JButton trainButton = new JButton("Train");
		trainButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				trainButton.setEnabled(false);

				new Thread() {
					public void run() {
						driver.network.train(driver.X, driver.Y, NUM_CLASSES,
								LAMBDA, listener, true);

						SwingUtilities.invokeLater(new Runnable() {
							public void run() {
								trainButton.setEnabled(true);
							}
						});
					}
				}.start();
			}
		});
		sidebar.add(trainButton);

		costLabel = new JLabel("Cost: 0.0000");
		sidebar.add(costLabel);

		return sidebar;
	}

	/**
	 * Display the result of a prediction.
	 */
	private void updateSidebar(double[] outputUnits, int predictedClass,
			long nanoseconds) {
		long microseconds = nanoseconds / 1000L;
		predictionLabel.setText(classLabels[predictedClass] + " ("
				+ microseconds + " micros)");

		for ( int i = 0; i < outputUnits.length; i++ ) {
			classBars[i].setValue((int) (outputUnits[i] * 1000));
		}
	}

	// private void nextExample() {
	// // Pick a random example
	// int i = (int) Math.round(Math.random() * X.getRows());
	//
	// // Build a DoubleMatrix row vector
	// DoubleMatrix x = X.getRow(i);
	//
	// // Benchmark
	// long start = System.nanoTime();
	//
	// // Get the output layer (just a column vector)
	// ForwardPropagationResult fResult = network.feedForward(x);
	// double[] units = fResult.getOutputLayer().getColumn(0).toArray();
	// int predictedClass = NeuralNetwork.maxIndex(units);
	// long end = System.nanoTime();
	//
	// updateSidebar(units, predictedClass, end - start);
	//
	// // Show
	// DoubleMatrix image = normalize(MatrixTools.reshape(x.toArray()));
	// applet.setImage(image);
	// }

	private void loadImageMatrix(DoubleMatrix m) {
		// TODO: Forward prop

		applet.setImage(m);
	}

	public static void main(String[] args) {
		new NeuralNetworkDriverGUI().init();
	}

}
