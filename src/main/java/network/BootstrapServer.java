package network;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import java.io.*;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The Federated Learning Hub.
 * Handles peer discovery and hosts training/testing datasets for active models.
 */
public class BootstrapServer {

    private static final Map<String, List<Peer>> modelPeers = new ConcurrentHashMap<>();
    private static final Map<String, ModelConfig> activeModels = new ConcurrentHashMap<>();
    private static final Gson gson = new Gson();
    
    // Directory to store uploaded datasets locally
    private static final Path DATA_DIR = Paths.get("bootstrap_data");

    public static void main(String[] args) throws IOException {
        Files.createDirectories(DATA_DIR);

        int port = 9000;
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);

        server.createContext("/join", new JoinHandler());
        server.createContext("/models", new ModelsHandler());
        server.createContext("/registerModel", new RegisterModelHandler());
        server.createContext("/data", new DataDownloadHandler());
        
        server.setExecutor(java.util.concurrent.Executors.newCachedThreadPool());
        server.start();
        
        System.out.println("=== FedChain Hub (Bootstrap Server) ===");
        System.out.println("Listening for connections on port " + port);
        System.out.println("Data storage: " + DATA_DIR.toAbsolutePath());
    }

    // --- 1. JOIN (Peer Discovery) ---
    static class JoinHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(405, -1); return;
            }
            
            // Parse query params (modelId, nodeId, ip, port)
            Map<String, String> params = parseQuery(exchange.getRequestURI().getQuery());
            if (!params.containsKey("modelId") || !params.containsKey("nodeId") || 
                !params.containsKey("ip") || !params.containsKey("port")) {
                sendError(exchange, 400, "Missing parameters");
                return;
            }

            Peer newPeer = new Peer(params.get("nodeId"), params.get("ip"), Integer.parseInt(params.get("port")));
            String modelId = params.get("modelId");

            List<Peer> existingPeers = modelPeers.computeIfAbsent(modelId, k -> Collections.synchronizedList(new ArrayList<>()));
            List<Peer> peersToReturn = new ArrayList<>(existingPeers);
            
            existingPeers.removeIf(p -> p.getNodeId().equals(newPeer.getNodeId()));
            existingPeers.add(newPeer);

            System.out.println("[JOIN] Node " + newPeer.getNodeId() + " joined model '" + modelId + "'");
            sendJsonResponse(exchange, 200, gson.toJson(peersToReturn));
        }
    }

    // --- 2. GET MODELS ---
    static class ModelsHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(405, -1); return;
            }
            // Return list of all active configurations
            sendJsonResponse(exchange, 200, gson.toJson(activeModels.values()));
        }
    }

    // --- 3. REGISTER NEW MODEL ---
    static class RegisterModelHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"POST".equals(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(405, -1); return;
            }

            // Read the JSON body containing config + Base64 encoded datasets
            InputStream is = exchange.getRequestBody();
            String body = new String(is.readAllBytes(), StandardCharsets.UTF_8);
            
            // Parse the wrapper object
            JsonObject payload = gson.fromJson(body, JsonObject.class);
            ModelConfig config = gson.fromJson(payload.get("config"), ModelConfig.class);
            
            String trainImg = payload.get("trainImages").getAsString();
            String trainLbl = payload.get("trainLabels").getAsString();
            String testImg = payload.get("testImages").getAsString();
            String testLbl = payload.get("testLabels").getAsString();

            // Store datasets locally
            Path modelDir = DATA_DIR.resolve(config.modelId);
            Files.createDirectories(modelDir);
            
            Files.write(modelDir.resolve("trainImages.gz"), Base64.getDecoder().decode(trainImg));
            Files.write(modelDir.resolve("trainLabels.gz"), Base64.getDecoder().decode(trainLbl));
            Files.write(modelDir.resolve("testImages.gz"), Base64.getDecoder().decode(testImg));
            Files.write(modelDir.resolve("testLabels.gz"), Base64.getDecoder().decode(testLbl));

            activeModels.put(config.modelId, config);
            System.out.println("[REGISTER] New model created: " + config.modelId);

            sendJsonResponse(exchange, 200, "{\"status\":\"Success\"}");
        }
    }

    // --- 4. DOWNLOAD DATA ---
    static class DataDownloadHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            if (!"GET".equals(exchange.getRequestMethod())) {
                exchange.sendResponseHeaders(405, -1); return;
            }
            
            Map<String, String> params = parseQuery(exchange.getRequestURI().getQuery());
            String modelId = params.get("modelId");
            String type = params.get("type"); // "trainImages.gz", etc.

            if (modelId == null || type == null) {
                sendError(exchange, 400, "Missing modelId or type");
                return;
            }

            Path file = DATA_DIR.resolve(modelId).resolve(type);
            if (!Files.exists(file)) {
                sendError(exchange, 404, "Data not found");
                return;
            }

            byte[] data = Files.readAllBytes(file);
            exchange.getResponseHeaders().set("Content-Type", "application/zip");
            exchange.sendResponseHeaders(200, data.length);
            OutputStream os = exchange.getResponseBody();
            os.write(data);
            os.close();
            System.out.println("[DATA] Served " + type + " data for model " + modelId);
        }
    }

    // --- Helpers ---
    private static void sendJsonResponse(HttpExchange exchange, int code, String json) throws IOException {
        byte[] bytes = json.getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().set("Content-Type", "application/json");
        exchange.sendResponseHeaders(code, bytes.length);
        OutputStream os = exchange.getResponseBody();
        os.write(bytes);
        os.close();
    }

    private static void sendError(HttpExchange exchange, int code, String message) throws IOException {
        String errorJson = "{\"error\":\"" + message + "\"}";
        sendJsonResponse(exchange, code, errorJson);
    }

    private static Map<String, String> parseQuery(String query) {
        Map<String, String> map = new HashMap<>();
        if (query == null) return map;
        for (String param : query.split("&")) {
            String[] kv = param.split("=");
            if (kv.length == 2) map.put(kv[0], kv[1]);
        }
        return map;
    }
}
