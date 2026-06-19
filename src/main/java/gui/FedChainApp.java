package gui;

import blockchain.BlockChain;
import com.google.gson.*;
import com.google.gson.reflect.TypeToken;
import math.Matrix;
import ml.MNISTLoader;
import ml.NeuralNetwork;
import network.ModelConfig;
import network.NodeLauncher;
import network.Peer;

import javax.swing.*;
import java.awt.*;
import java.io.*;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.UUID;

public class FedChainApp extends JFrame {

    private final String bootstrapUrl = "http://localhost:9000";
    private JTextArea logArea;
    private final Gson gson = new Gson();
    
    // File references for upload
    private File trainImagesFile, trainLabelsFile, testImagesFile, testLabelsFile;
    private NodeLauncher activeNode;

    public FedChainApp() {
        setTitle("FedChain P2P Node");
        setSize(900, 650);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);

        JTabbedPane tabbedPane = new JTabbedPane();
        tabbedPane.addTab("Join Network", createJoinPanel());
        tabbedPane.addTab("Create New Model", createModelPanel());
        tabbedPane.addTab("Live Dashboard", createDashboardPanel());
        add(tabbedPane);
        
        setGlobalFont(this, new Font("SansSerif", Font.PLAIN, 14));
        redirectSystemOut();
    }

    private void redirectSystemOut() {
        OutputStream out = new OutputStream() {
            @Override
            public void write(int b) {
                SwingUtilities.invokeLater(() -> {
                    logArea.append(String.valueOf((char) b));
                    logArea.setCaretPosition(logArea.getDocument().getLength());
                });
            }
        };
        System.setOut(new PrintStream(out, true));
    }

    private JPanel createJoinPanel() {
        JPanel panel = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.insets = new Insets(10, 10, 10, 10);
        gbc.fill = GridBagConstraints.HORIZONTAL;

        JButton refreshModelsBtn = new JButton("Fetch Active Models");
        JComboBox<String> modelDropdown = new JComboBox<>();
        JComboBox<String> roleDropdown = new JComboBox<>(new String[]{"TRAINER", "VALIDATOR"});
        JButton joinBtn = new JButton("Join Network & Start Node");

        refreshModelsBtn.addActionListener(e -> {
            new Thread(() -> {
                try {
                    String response = httpGet(bootstrapUrl + "/models");
                    JsonArray models = JsonParser.parseString(response).getAsJsonArray();
                    SwingUtilities.invokeLater(() -> {
                        modelDropdown.removeAllItems();
                        for (JsonElement el : models) {
                            modelDropdown.addItem(el.getAsJsonObject().get("modelId").getAsString());
                        }
                        System.out.println("Fetched " + models.size() + " active models.");
                    });
                } catch (Exception ex) {
                    System.out.println("Error fetching models: " + ex.getMessage());
                }
            }).start();
        });

        joinBtn.addActionListener(e -> {
            String roleStr = (String) roleDropdown.getSelectedItem();
            String modelId = (String) modelDropdown.getSelectedItem();
            if (modelId == null) {
                System.out.println("Please select a model first.");
                return;
            }
            joinBtn.setEnabled(false);
            
            new Thread(() -> {
                try {
                    System.out.println("Fetching node data from hub for " + modelId + "...");
                    NodeLauncher.Role role = NodeLauncher.Role.valueOf(roleStr);
                    
                    Path tempDir = Files.createTempDirectory("fedchain_" + UUID.randomUUID());
                    Matrix[] images, labels;

                    if (role == NodeLauncher.Role.TRAINER) {
                        downloadFile(modelId, "trainImages.gz", tempDir.resolve("images.gz"));
                        downloadFile(modelId, "trainLabels.gz", tempDir.resolve("labels.gz"));
                    } else {
                        downloadFile(modelId, "testImages.gz", tempDir.resolve("images.gz"));
                        downloadFile(modelId, "testLabels.gz", tempDir.resolve("labels.gz"));
                    }

                    System.out.println("Loading downloaded data into memory...");
                    images = MNISTLoader.loadImages(tempDir.resolve("images.gz").toString());
                    labels = MNISTLoader.loadLabels(tempDir.resolve("labels.gz").toString());

                    String myNodeId = "Node_" + UUID.randomUUID().toString().substring(0, 5);
                    int myPort = 8000 + (int)(Math.random() * 1000); // Random port for demo

                    System.out.println("Fetching peers from hub...");
                    String peersJson = httpGet(bootstrapUrl + "/join?modelId=" + modelId + "&nodeId=" + myNodeId + "&ip=127.0.0.1&port=" + myPort);
                    List<Peer> initialPeers = gson.fromJson(peersJson, new TypeToken<List<Peer>>(){}.getType());

                    System.out.println("Initializing NodeLauncher...");
                    
                    // Note: We need to dynamically fetch the ModelConfig in a real scenario to get architecture array
                    // For this prototype, we'll assume standard architecture
                    NeuralNetwork globalModel = new NeuralNetwork(new int[]{784, 128, 64, 10}, 0.01, 0L);
                    
                    activeNode = new NodeLauncher(
                        myNodeId, myPort, role, initialPeers, globalModel, new BlockChain(5.0),
                        images, labels, 3, 2
                    );
                    
                    activeNode.start();
                    System.out.println("NODE SUCCESSFULLY JOINED THE NETWORK!");

                } catch (Exception ex) {
                    System.out.println("Error joining network: " + ex.getMessage());
                    ex.printStackTrace();
                    SwingUtilities.invokeLater(() -> joinBtn.setEnabled(true));
                }
            }).start();
        });

        gbc.gridx = 0; gbc.gridy = 0; panel.add(refreshModelsBtn, gbc);
        gbc.gridx = 1; gbc.gridy = 0; panel.add(modelDropdown, gbc);
        gbc.gridx = 0; gbc.gridy = 1; panel.add(new JLabel("Role:"), gbc);
        gbc.gridx = 1; gbc.gridy = 1; panel.add(roleDropdown, gbc);
        gbc.gridx = 0; gbc.gridy = 2; gbc.gridwidth = 2; panel.add(joinBtn, gbc);

        return panel;
    }

    private void downloadFile(String modelId, String type, Path dest) throws Exception {
        URL url = new URL(bootstrapUrl + "/data?modelId=" + modelId + "&type=" + type);
        try (InputStream in = url.openStream()) {
            Files.copy(in, dest);
        }
    }

    private JPanel createModelPanel() {
        JPanel panel = new JPanel(new GridLayout(7, 2, 10, 10));
        panel.setBorder(BorderFactory.createEmptyBorder(20, 20, 20, 20));

        JTextField modelIdField = new JTextField("mnist-run-" + UUID.randomUUID().toString().substring(0, 4));
        JTextField archField = new JTextField("784,128,64,10");
        
        JButton btnTrainImg = new JButton("Select Train Images (.gz)");
        JButton btnTrainLbl = new JButton("Select Train Labels (.gz)");
        JButton btnTestImg = new JButton("Select Test Images (.gz)");
        JButton btnTestLbl = new JButton("Select Test Labels (.gz)");
        
        JButton submitBtn = new JButton("Upload & Register Model");

        JFileChooser fileChooser = new JFileChooser(new File("data/mnist"));

        btnTrainImg.addActionListener(e -> { if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) { trainImagesFile = fileChooser.getSelectedFile(); System.out.println("Selected " + trainImagesFile.getName()); }});
        btnTrainLbl.addActionListener(e -> { if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) { trainLabelsFile = fileChooser.getSelectedFile(); System.out.println("Selected " + trainLabelsFile.getName()); }});
        btnTestImg.addActionListener(e -> { if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) { testImagesFile = fileChooser.getSelectedFile(); System.out.println("Selected " + testImagesFile.getName()); }});
        btnTestLbl.addActionListener(e -> { if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) { testLabelsFile = fileChooser.getSelectedFile(); System.out.println("Selected " + testLabelsFile.getName()); }});

        submitBtn.addActionListener(e -> {
            if (trainImagesFile == null || trainLabelsFile == null || testImagesFile == null || testLabelsFile == null) {
                System.out.println("Please select all 4 dataset files first!");
                return;
            }
            submitBtn.setEnabled(false);
            
            new Thread(() -> {
                try {
                    System.out.println("Encoding dataset to Base64 (this may take a moment)...");
                    JsonObject payload = new JsonObject();
                    
                    String[] archStrs = archField.getText().split(",");
                    int[] arch = new int[archStrs.length];
                    for(int i=0; i<arch.length; i++) arch[i] = Integer.parseInt(archStrs[i].trim());
                    
                    ModelConfig config = new ModelConfig(modelIdField.getText(), arch, 0.01, 3, 2, 1.0);
                    payload.add("config", gson.toJsonTree(config));
                    
                    payload.addProperty("trainImages", encodeFile(trainImagesFile));
                    payload.addProperty("trainLabels", encodeFile(trainLabelsFile));
                    payload.addProperty("testImages", encodeFile(testImagesFile));
                    payload.addProperty("testLabels", encodeFile(testLabelsFile));
                    
                    System.out.println("Uploading payload to Bootstrap Server...");
                    httpPost(bootstrapUrl + "/registerModel", payload.toString());
                    
                    System.out.println("SUCCESS: Model Registered!");
                } catch (Exception ex) {
                    System.out.println("Upload failed: " + ex.getMessage());
                } finally {
                    SwingUtilities.invokeLater(() -> submitBtn.setEnabled(true));
                }
            }).start();
        });

        panel.add(new JLabel("Model ID:")); panel.add(modelIdField);
        panel.add(new JLabel("Architecture:")); panel.add(archField);
        panel.add(btnTrainImg); panel.add(btnTrainLbl);
        panel.add(btnTestImg); panel.add(btnTestLbl);
        panel.add(new JLabel("")); panel.add(submitBtn);

        return panel;
    }

    private String encodeFile(File file) throws Exception {
        byte[] bytes = Files.readAllBytes(file.toPath());
        return Base64.getEncoder().encodeToString(bytes);
    }

    private JPanel createDashboardPanel() {
        JPanel panel = new JPanel(new BorderLayout());
        logArea = new JTextArea();
        logArea.setEditable(false);
        logArea.setBackground(Color.BLACK);
        logArea.setForeground(Color.GREEN);
        logArea.setFont(new Font("Monospaced", Font.PLAIN, 14));
        panel.add(new JScrollPane(logArea), BorderLayout.CENTER);
        return panel;
    }

    private String httpGet(String urlStr) throws Exception {
        URL url = new URL(urlStr);
        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setRequestMethod("GET");
        try(BufferedReader in = new BufferedReader(new InputStreamReader(con.getInputStream()))) {
            StringBuilder content = new StringBuilder();
            String inputLine;
            while ((inputLine = in.readLine()) != null) content.append(inputLine);
            return content.toString();
        }
    }

    private void httpPost(String urlStr, String jsonBody) throws Exception {
        URL url = new URL(urlStr);
        HttpURLConnection con = (HttpURLConnection) url.openConnection();
        con.setRequestMethod("POST");
        con.setRequestProperty("Content-Type", "application/json");
        con.setDoOutput(true);
        try(OutputStream os = con.getOutputStream()) {
            os.write(jsonBody.getBytes(StandardCharsets.UTF_8));
        }
        if (con.getResponseCode() != 200) throw new Exception("HTTP Error: " + con.getResponseCode());
    }

    private static void setGlobalFont(Component comp, Font font) {
        if (!(comp instanceof JTextArea)) comp.setFont(font);
        if (comp instanceof Container) {
            for (Component child : ((Container) comp).getComponents()) setGlobalFont(child, font);
        }
    }

    public static void main(String[] args) {
        System.setProperty("awt.useSystemAAFontSettings", "on");
        System.setProperty("swing.aatext", "true");
        try {
            for (UIManager.LookAndFeelInfo info : UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    UIManager.setLookAndFeel(info.getClassName()); break;
                }
            }
        } catch (Exception ignored) {}
        SwingUtilities.invokeLater(() -> new FedChainApp().setVisible(true));
    }
}
