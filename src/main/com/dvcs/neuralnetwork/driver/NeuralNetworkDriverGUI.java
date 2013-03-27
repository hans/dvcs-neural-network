package com.dvcs.neuralnetwork.driver;

import java.awt.Font;
import java.awt.GridLayout;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.logging.Logger;

import javax.swing.BoxLayout;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.SwingConstants;
import javax.swing.SwingUtilities;

import org.jblas.DoubleMatrix;

import de.jungblut.math.DoubleVector;
import de.jungblut.math.minimize.MinimizerListener;

public class NeuralNetworkDriverGUI {

	static final int NUM_CLASSES = 10;
	static final String[] classLabels = new String[] { "1", "2", "3", "4", "5",
			"6", "7", "8", "9", "0" };

	NeuralNetworkDriver driver;

	JFrame frame;
	MatrixImageApplet applet;

	JPanel sidebar;
	JProgressBar[] classBars;
	JLabel predictionLabel;

	JPanel trainingSidebar;
	JLabel costLabel;

	public void init() {
		// Debug all
		LogManager.getLogManager().getLogger(Logger.GLOBAL_LOGGER_NAME)
				.setLevel(Level.FINEST);

		driver = new NeuralNetworkDriver(this);

		frame = new JFrame();
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.setSize(1010, 400);
		frame.getContentPane().setLayout(
				new BoxLayout(frame.getContentPane(), BoxLayout.Y_AXIS));

		JPanel statusPanel = initStatusPanel();
		frame.add(statusPanel);

		JPanel adminBar = initAdminBar();
		frame.add(adminBar);

		JPanel mainPanel = initMainPanel();
		frame.add(mainPanel);

		frame.setVisible(true);
	}

	private JPanel initStatusPanel() {
		JPanel p = new JPanel();

		// TODO

		return p;
	}

	private JPanel initAdminBar() {
		JPanel adminBar = new JPanel();
		adminBar.setLayout(new BoxLayout(adminBar, BoxLayout.X_AXIS));

		final JButton startListeningButton = new JButton("Start collecting");
		startListeningButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				driver.startCollecting();
				startListeningButton.setEnabled(false);
			}
		});
		adminBar.add(startListeningButton);

		return adminBar;
	}

	private JPanel initMainPanel() {
		JPanel p = new JPanel();
		p.setLayout(new BoxLayout(p, BoxLayout.X_AXIS));

		applet = new MatrixImageApplet();
		applet.init();
		applet.start();
		p.add(applet);

		JPanel sidebar = initSidebar();
		p.add(sidebar);

		return p;
	}

	private JPanel initSidebar() {
		JPanel sidebar = new JPanel();

		GridLayout layout = new GridLayout(0, 2);
		sidebar.setLayout(layout);

		JLabel placeholderButton = new JLabel("");
		sidebar.add(placeholderButton);

		predictionLabel = new JLabel();
		predictionLabel.setFont(new Font("Sans Serif", Font.PLAIN, 18));
		predictionLabel.setHorizontalAlignment(SwingConstants.CENTER);
		sidebar.add(predictionLabel);

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
						driver.trainNeuralNetwork();

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

	void loadImageMatrix(DoubleMatrix m) {
		// TODO: Forward prop

		applet.setImage(m);
	}

	public static void main(String[] args) {
		new NeuralNetworkDriverGUI().init();
	}

}
