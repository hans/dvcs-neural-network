package com.dvcs.neuralnetwork.driver;

import java.awt.Font;
import java.awt.GridLayout;
import java.awt.KeyEventDispatcher;
import java.awt.KeyboardFocusManager;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.KeyEvent;
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

public class DriverGUI implements KeyEventDispatcher {

	static final int NUM_CLASSES = 3;
	static final String[] classLabels = new String[] { "Left", "Right", "Up" };

	Driver driver;
	DriverOutputManager outputManager;

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

		outputManager = new DriverOutputManager();
		driver = new Driver(this, outputManager);

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

		KeyboardFocusManager.getCurrentKeyboardFocusManager()
				.addKeyEventDispatcher(this);
	}

	private JPanel initStatusPanel() {
		JPanel p = new JPanel();

		// TODO

		return p;
	}

	private JPanel initAdminBar() {
		JPanel adminBar = new JPanel();
		adminBar.setLayout(new BoxLayout(adminBar, BoxLayout.X_AXIS));

		final JButton listeningToggleButton = new JButton("Start collecting");

		final JButton buildNetworkButton = new JButton("Build neural network");
		buildNetworkButton.setEnabled(false);

		final JButton feedForwardButton = new JButton("Start feedforward");
		feedForwardButton.setEnabled(false);

		listeningToggleButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				if ( driver.isCollecting() ) {
					driver.stopCollecting();
					listeningToggleButton.setText("Start collecting");

					if ( driver.hasSufficientData() ) {
						buildNetworkButton.setEnabled(true);
					} else {
						buildNetworkButton.setEnabled(false);
					}
				} else {
					driver.startCollecting();
					listeningToggleButton.setText("Stop collecting");
				}
			}
		});

		buildNetworkButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				driver.trainNeuralNetwork();
				feedForwardButton.setEnabled(true);
			}
		});

		feedForwardButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {
				driver.startFeedForward();
			}
		});

		adminBar.add(listeningToggleButton);
		adminBar.add(buildNetworkButton);
		adminBar.add(feedForwardButton);

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

		updateClassBars(outputUnits);
	}
	
	private void updateClassBars(double[] outputUnits) {
		for ( int i = 0; i < outputUnits.length; i++ ) {
			classBars[i].setValue((int) (outputUnits[i] * 1000));
		}
	}

	void displayPropagationResult(double[] outputUnits, int predictedClass,
			long time) {
		updateSidebar(outputUnits, predictedClass, time);
	}

	void loadImageMatrix(DoubleMatrix m) {
		// TODO: Forward prop

		applet.setImage(m);
	}

	public static void main(String[] args) {
		new DriverGUI().init();
	}

	@Override
	public boolean dispatchKeyEvent(KeyEvent e) {
		boolean enabled = e.getID() == KeyEvent.KEY_PRESSED;

		switch ( e.getKeyCode() ) {
		case KeyEvent.VK_LEFT:
			outputManager.setLeftArrowEnabled(enabled);
			break;
		case KeyEvent.VK_RIGHT:
			outputManager.setRightArrowEnabled(enabled);
			break;
		case KeyEvent.VK_UP:
			outputManager.setUpArrowEnabled(enabled);
			break;
		}
		
		updateClassBars(outputManager.getOutput());

		return true;
	}

}
