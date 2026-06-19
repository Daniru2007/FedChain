package gui;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;

/**
 * Main Client Entry Point for FedChain.
 * A Swing GUI that allows users to create models, join networks, and monitor progress.
 */
public class FedChainApp extends JFrame {

    private final String bootstrapUrl = "http://localhost:9000";
    private JTextArea logArea;

    public FedChainApp() {
        setTitle("FedChain P2P Node");
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        JTabbedPane tabbedPane = new JTabbedPane();

        // Tabs
        tabbedPane.addTab("Join Network", createJoinPanel());
        tabbedPane.addTab("Create New Model", createModelPanel());
        tabbedPane.addTab("Live Dashboard", createDashboardPanel());

        add(tabbedPane);
    }

    private JPanel createJoinPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(10, 10, 10, 10);
        gbc.fill = GridBagConstraints.HORIZONTAL;

        JButton refreshModelsBtn = new JButton("Fetch Active Models");
        JComboBox<String> modelDropdown = new JComboBox<>();
        
        JComboBox<String> roleDropdown = new JComboBox<>(new String[]{"TRAINER", "VALIDATOR"});
        
        JButton joinBtn = new JButton("Join Network & Download Data");

        refreshModelsBtn.addActionListener(e -> {
            // In reality, this makes a GET request to /models and parses JSON
            modelDropdown.removeAllItems();
            modelDropdown.addItem("mnist-run-001");
            log("Fetched active models from Hub.");
        });

        joinBtn.addActionListener(e -> {
            String role = (String) roleDropdown.getSelectedItem();
            String model = (String) modelDropdown.getSelectedItem();
            if (model == null) return;
            
            log("Connecting to Bootstrap Server for " + model + "...");
            log("Role: " + role + ". Downloading dataset...");
            
            // Here we would:
            // 1. Download the ZIP file via GET /data
            // 2. Unpack the ZIP
            // 3. GET /join to fetch initial peers
            // 4. Start the NodeLauncher
            
            joinBtn.setEnabled(false);
            log("Dataset downloaded. Node networking started.");
        });

        gbc.gridx = 0; gbc.gridy = 0; panel.add(refreshModelsBtn, gbc);
        gbc.gridx = 1; gbc.gridy = 0; panel.add(modelDropdown, gbc);
        
        gbc.gridx = 0; gbc.gridy = 1; panel.add(new JLabel("Role:"), gbc);
        gbc.gridx = 1; gbc.gridy = 1; panel.add(roleDropdown, gbc);
        
        gbc.gridx = 0; gbc.gridy = 2; gbc.gridwidth = 2; 
        panel.add(joinBtn, gbc);

        return panel;
    }

    private JPanel createModelPanel() {
        JPanel panel = new JPanel(new GridLayout(6, 2, 10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        JTextField modelIdField = new JTextField("mnist-run-001");
        JTextField archField = new JTextField("784,128,64,10");
        
        JButton selectTrainBtn = new JButton("Select Train ZIP");
        JButton selectTestBtn = new JButton("Select Test ZIP");
        
        JButton submitBtn = new JButton("Upload & Register Model");

        selectTrainBtn.addActionListener(e -> log("Selected Training Data"));
        selectTestBtn.addActionListener(e -> log("Selected Testing Data"));

        submitBtn.addActionListener(e -> {
            log("Packaging architecture and encoding datasets to Base64...");
            log("Uploading to Bootstrap Server via POST /registerModel...");
            // Simulate upload delay
            new Thread(() -> {
                try { Thread.sleep(1000); } catch (InterruptedException ex) {}
                SwingUtilities.invokeLater(() -> log("Model successfully registered! It is now live on the network."));
            }).start();
        });

        panel.add(new JLabel("Model ID:")); panel.add(modelIdField);
        panel.add(new JLabel("Architecture:")); panel.add(archField);
        panel.add(new JLabel("Training Data:")); panel.add(selectTrainBtn);
        panel.add(new JLabel("Testing Data:")); panel.add(selectTestBtn);
        panel.add(new JLabel("")); panel.add(submitBtn);

        return panel;
    }

    private JPanel createDashboardPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        
        logArea = new JTextArea();
        logArea.setEditable(false);
        logArea.setBackground(Color.BLACK);
        logArea.setForeground(Color.GREEN);
        logArea.setFont(new Font("Monospaced", Font.PLAIN, 12));
        
        JScrollPane scrollPane = new JScrollPane(logArea);
        
        JPanel statsPanel = new JPanel(new GridLayout(1, 3));
        statsPanel.add(new JLabel("  Node Status: ONLINE  "));
        statsPanel.add(new JLabel("  Peers: 0  "));
        statsPanel.add(new JLabel("  Global Accuracy: --%  "));
        
        panel.add(statsPanel, BorderLayout.NORTH);
        panel.add(scrollPane, BorderLayout.CENTER);

        return panel;
    }

    private void log(String message) {
        if (logArea != null) {
            SwingUtilities.invokeLater(() -> logArea.append("> " + message + "\n"));
        }
    }

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {}

        SwingUtilities.invokeLater(() -> {
            new FedChainApp().setVisible(true);
        });
    }
}
