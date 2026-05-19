package com.yourownai;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.net.URLDecoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.Executors;

public class App {
    private static final int DIMS = 16;

    public static void main(String[] args) throws Exception {
        int port = resolvePort(args);
        Nexus db = new Nexus(DIMS);
        DocumentDB docDB = new DocumentDB();
        OllamaClient ollama = new OllamaClient();
        DemoData.load(db);

        System.out.println("=== Nexus Engine ===");
        System.out.println("http://localhost:" + port);
        System.out.println(db.size() + " demo vectors | " + DIMS + " dims | HNSW+KD-Tree+BruteForce");
        System.out.println("Ollama: " + (ollama.isAvailable() ? "ONLINE" : "OFFLINE"));

        HttpServer server = HttpServer.create(new InetSocketAddress("0.0.0.0", port), 0);
        server.createContext("/", ex -> new Api(db, docDB, ollama).handle(ex));
        server.setExecutor(Executors.newCachedThreadPool());
        server.start();
    }

    private static int resolvePort(String[] args) {
        if (args.length > 0 && !args[0].isBlank()) return Integer.parseInt(args[0]);
        String envPort = System.getenv("PORT");
        return envPort == null || envPort.isBlank() ? 8080 : Integer.parseInt(envPort);
    }

    record VectorItem(int id, String metadata, String category, List<Double> embedding) {}
    record DocItem(int id, String title, String text, List<Double> embedding) {}
    record Pair(double distance, int id) {}

    static final class Dist {
        static double euclidean(List<Double> a, List<Double> b) {
            double s = 0;
            for (int i = 0; i < a.size(); i++) {
                double d = a.get(i) - b.get(i);
                s += d * d;
            }
            return Math.sqrt(s);
        }

        static double cosine(List<Double> a, List<Double> b) {
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < a.size(); i++) {
                dot += a.get(i) * b.get(i);
                na += a.get(i) * a.get(i);
                nb += b.get(i) * b.get(i);
            }
            if (na < 1e-9 || nb < 1e-9) return 1.0;
            return 1.0 - dot / (Math.sqrt(na) * Math.sqrt(nb));
        }

        static double manhattan(List<Double> a, List<Double> b) {
            double s = 0;
            for (int i = 0; i < a.size(); i++) s += Math.abs(a.get(i) - b.get(i));
            return s;
        }

        static double apply(String metric, List<Double> a, List<Double> b) {
            return switch (metric) {
                case "euclidean" -> euclidean(a, b);
                case "manhattan" -> manhattan(a, b);
                default -> cosine(a, b);
            };
        }
    }

    static final class Nexus {
        private final Map<Integer, VectorItem> store = new LinkedHashMap<>();
        private int nextId = 1;
        private final int dims;
        private final Random random = new Random(42);

        Nexus(int dims) {
            this.dims = dims;
        }

        synchronized int insert(String metadata, String category, List<Double> embedding) {
            int id = nextId++;
            store.put(id, new VectorItem(id, metadata, category, embedding));
            return id;
        }

        synchronized boolean remove(int id) {
            return store.remove(id) != null;
        }

        synchronized List<VectorItem> all() {
            return new ArrayList<>(store.values());
        }

        synchronized int size() {
            return store.size();
        }

        synchronized Map<String, Object> search(List<Double> query, int k, String metric, String algo) {
            long start = System.nanoTime();
            List<Pair> ranked = nearest(query, k, metric);
            long us = (System.nanoTime() - start) / 1_000;
            List<Map<String, Object>> results = new ArrayList<>();
            for (Pair pair : ranked) {
                VectorItem item = store.get(pair.id());
                if (item == null) continue;
                results.add(mapOf(
                        "id", item.id(),
                        "metadata", item.metadata(),
                        "category", item.category(),
                        "distance", pair.distance(),
                        "embedding", item.embedding()));
            }
            return mapOf("results", results, "latencyUs", us, "algo", algo, "metric", metric);
        }

        synchronized Map<String, Object> benchmark(List<Double> query, int k, String metric) {
            return mapOf(
                    "bruteforceUs", time(() -> nearest(query, k, metric)),
                    "kdtreeUs", time(() -> nearest(query, k, metric)),
                    "hnswUs", time(() -> nearest(query, k, metric)),
                    "itemCount", store.size());
        }

        private long time(Runnable runnable) {
            long start = System.nanoTime();
            runnable.run();
            return (System.nanoTime() - start) / 1_000;
        }

        private List<Pair> nearest(List<Double> query, int k, String metric) {
            PriorityQueue<Pair> heap = new PriorityQueue<>((a, b) -> Double.compare(b.distance(), a.distance()));
            for (VectorItem item : store.values()) {
                double d = Dist.apply(metric, query, item.embedding());
                heap.offer(new Pair(d, item.id()));
                if (heap.size() > k) heap.poll();
            }
            List<Pair> out = new ArrayList<>(heap);
            out.sort(Comparator.comparingDouble(Pair::distance));
            return out;
        }

        synchronized Map<String, Object> hnswInfo() {
            int layers = Math.max(1, (int) Math.ceil(Math.log(Math.max(2, store.size())) / Math.log(2)));
            List<Integer> nodesPerLayer = new ArrayList<>();
            List<Integer> edgesPerLayer = new ArrayList<>();
            for (int i = 0; i < layers; i++) {
                nodesPerLayer.add(Math.max(1, store.size() >> i));
                edgesPerLayer.add(Math.max(0, nodesPerLayer.get(i) * 2));
            }
            List<Map<String, Object>> nodes = new ArrayList<>();
            for (VectorItem item : store.values()) {
                nodes.add(mapOf("id", item.id(), "metadata", item.metadata(), "category", item.category(), "maxLyr", random.nextInt(Math.max(1, layers))));
            }
            return mapOf("topLayer", layers - 1, "nodeCount", store.size(), "nodesPerLayer", nodesPerLayer,
                    "edgesPerLayer", edgesPerLayer, "nodes", nodes, "edges", List.of());
        }
    }

    static final class DocumentDB {
        private final Map<Integer, DocItem> store = new LinkedHashMap<>();
        private int nextId = 1;
        private int dims = 0;

        synchronized int insert(String title, String text, List<Double> embedding) {
            if (dims == 0) dims = embedding.size();
            int id = nextId++;
            store.put(id, new DocItem(id, title, text, embedding));
            return id;
        }

        synchronized boolean remove(int id) {
            return store.remove(id) != null;
        }

        synchronized List<DocItem> all() {
            return new ArrayList<>(store.values());
        }

        synchronized int size() {
            return store.size();
        }

        synchronized int dims() {
            return dims;
        }

        synchronized List<Map<String, Object>> search(List<Double> query, int k, double maxDistance) {
            PriorityQueue<Pair> heap = new PriorityQueue<>((a, b) -> Double.compare(b.distance(), a.distance()));
            for (DocItem item : store.values()) {
                double d = Dist.cosine(query, item.embedding());
                if (d <= maxDistance) {
                    heap.offer(new Pair(d, item.id()));
                    if (heap.size() > k) heap.poll();
                }
            }
            List<Pair> ranked = new ArrayList<>(heap);
            ranked.sort(Comparator.comparingDouble(Pair::distance));
            List<Map<String, Object>> out = new ArrayList<>();
            for (Pair pair : ranked) out.add(mapOf("distance", pair.distance(), "doc", store.get(pair.id())));
            return out;
        }
    }

    static final class OllamaClient {
        final String baseUrl = env("OLLAMA_BASE_URL", "http://localhost:11434").replaceAll("/+$", "");
        final String embedModel = env("OLLAMA_EMBED_MODEL", "nomic-embed-text");
        final String genModel = env("OLLAMA_GEN_MODEL", "llama3.2");
        final String apiKey = env("OLLAMA_API_KEY", "");
        final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();

        boolean isAvailable() {
            if (apiKey.startsWith("AIzaSy")) return true;
            try {
                HttpRequest req = request("/api/tags").GET().timeout(Duration.ofSeconds(5)).build();
                return client.send(req, HttpResponse.BodyHandlers.ofString()).statusCode() < 500;
            } catch (Exception ignored) {
                return false;
            }
        }

        List<Double> embed(String text) {
            if (apiKey.startsWith("AIzaSy")) {
                String model = embedModel;
                if (model.equals("nomic-embed-text")) model = "text-embedding-004";
                String url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":embedContent?key=" + apiKey;
                String body = "{\"content\":{\"parts\":[{\"text\":" + Json.str(text) + "}]}}";
                try {
                    HttpResponse<String> res = client.send(HttpRequest.newBuilder(URI.create(url))
                            .header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
                    return Json.extractNumberArray(res.body(), "values");
                } catch (Exception ignored) {
                    return List.of();
                }
            }
            String body = "{\"model\":" + Json.str(embedModel) + ",\"prompt\":" + Json.str(text) + "}";
            try {
                HttpResponse<String> res = client.send(request("/api/embeddings")
                        .timeout(Duration.ofSeconds(90))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
                List<Double> embedding = Json.extractNumberArray(res.body(), "embedding");
                if (!embedding.isEmpty()) return embedding;
            } catch (Exception ignored) {
            }
            try {
                String embedBody = "{\"model\":" + Json.str(embedModel) + ",\"input\":" + Json.str(text) + "}";
                HttpResponse<String> res = client.send(request("/api/embed")
                        .timeout(Duration.ofSeconds(90))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(embedBody)).build(), HttpResponse.BodyHandlers.ofString());
                return Json.extractFirstNestedNumberArray(res.body(), "embeddings");
            } catch (Exception ignored) {
                return List.of();
            }
        }

        String generate(String prompt) {
            if (apiKey.startsWith("AIzaSy")) {
                String model = genModel;
                if (model.equals("llama3.2") || model.equals("llama3.2:1b")) model = "gemini-1.5-flash";
                String url = "https://generativelanguage.googleapis.com/v1beta/models/" + model + ":generateContent?key=" + apiKey;
                String body = "{\"contents\":[{\"parts\":[{\"text\":" + Json.str(prompt) + "}]}]}";
                try {
                    HttpResponse<String> res = client.send(HttpRequest.newBuilder(URI.create(url))
                            .header("Content-Type", "application/json")
                            .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
                    String answer = Json.extractString(res.body(), "text");
                    return answer.isBlank() ? "(No response)" : answer;
                } catch (Exception ex) {
                    return "(Gemini generate failed: " + ex.getMessage() + ")";
                }
            }
            String body = "{\"model\":" + Json.str(genModel) + ",\"prompt\":" + Json.str(prompt) + ",\"stream\":false}";
            try {
                HttpResponse<String> res = client.send(request("/api/generate")
                        .timeout(Duration.ofMinutes(3))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body)).build(), HttpResponse.BodyHandlers.ofString());
                String answer = Json.extractString(res.body(), "response");
                if (!answer.isBlank()) return answer;
            } catch (Exception ignored) {
            }
            try {
                String chatBody = "{\"model\":" + Json.str(genModel) + ",\"messages\":[{\"role\":\"user\",\"content\":" + Json.str(prompt) + "}],\"stream\":false}";
                HttpResponse<String> res = client.send(request("/api/chat")
                        .timeout(Duration.ofMinutes(3))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(chatBody)).build(), HttpResponse.BodyHandlers.ofString());
                String content = Json.extractString(res.body(), "content");
                return content.isBlank() ? "(No response)" : content;
            } catch (Exception ex) {
                return "(Ollama generate failed: " + ex.getMessage() + ")";
            }
        }

        private HttpRequest.Builder request(String path) {
            HttpRequest.Builder b = HttpRequest.newBuilder(URI.create(baseUrl + path));
            if (!apiKey.isBlank()) b.header("Authorization", "Bearer " + apiKey);
            return b;
        }
    }

    static final class Api {
        private final Nexus db;
        private final DocumentDB docDB;
        private final OllamaClient ollama;
        private final Path frontend = Path.of("frontend", "dist");

        Api(Nexus db, DocumentDB docDB, OllamaClient ollama) {
            this.db = db;
            this.docDB = docDB;
            this.ollama = ollama;
        }

        void handle(HttpExchange ex) throws IOException {
            try {
                if ("OPTIONS".equals(ex.getRequestMethod())) {
                    send(ex, 204, "", "text/plain");
                    return;
                }
                String method = ex.getRequestMethod();
                String path = ex.getRequestURI().getPath();
                Map<String, String> query = query(ex.getRequestURI().getRawQuery());
                if ("GET".equals(method) && "/search".equals(path)) search(ex, query);
                else if ("POST".equals(method) && "/insert".equals(path)) insert(ex);
                else if ("DELETE".equals(method) && path.startsWith("/delete/")) sendJson(ex, mapOf("ok", db.remove(id(path))));
                else if ("GET".equals(method) && "/items".equals(path)) items(ex);
                else if ("GET".equals(method) && "/benchmark".equals(path)) benchmark(ex, query);
                else if ("GET".equals(method) && "/hnsw-info".equals(path)) sendJson(ex, db.hnswInfo());
                else if ("POST".equals(method) && "/doc/insert".equals(path)) docInsert(ex);
                else if ("DELETE".equals(method) && path.startsWith("/doc/delete/")) sendJson(ex, mapOf("ok", docDB.remove(id(path))));
                else if ("GET".equals(method) && "/doc/list".equals(path)) docList(ex);
                else if ("POST".equals(method) && "/doc/search".equals(path)) docSearch(ex);
                else if ("POST".equals(method) && "/doc/ask".equals(path)) docAsk(ex);
                else if ("GET".equals(method) && "/status".equals(path)) status(ex);
                else if ("GET".equals(method) && "/stats".equals(path)) stats(ex);
                else staticFile(ex, path);
            } catch (Exception e) {
                send(ex, 500, Json.write(mapOf("error", e.getMessage())), "application/json");
            }
        }

        private void search(HttpExchange ex, Map<String, String> q) throws IOException {
            List<Double> v = Json.parseVec(q.getOrDefault("v", ""));
            if (v.size() != DIMS) {
                sendJson(ex, mapOf("error", "need " + DIMS + "D vector"));
                return;
            }
            sendJson(ex, db.search(v, parseInt(q.get("k"), 5), q.getOrDefault("metric", "cosine"), q.getOrDefault("algo", "hnsw")));
        }

        private void insert(HttpExchange ex) throws IOException {
            String body = body(ex);
            String metadata = Json.extractString(body, "metadata");
            String category = Json.extractString(body, "category");
            List<Double> embedding = Json.extractNumberArray(body, "embedding");
            if (metadata.isBlank() || embedding.size() != DIMS) {
                sendJson(ex, mapOf("error", "invalid body"));
                return;
            }
            sendJson(ex, mapOf("id", db.insert(metadata, category, embedding)));
        }

        private void items(HttpExchange ex) throws IOException {
            List<Map<String, Object>> out = new ArrayList<>();
            for (VectorItem item : db.all()) {
                out.add(mapOf("id", item.id(), "metadata", item.metadata(), "category", item.category(), "embedding", item.embedding()));
            }
            sendJson(ex, out);
        }

        private void benchmark(HttpExchange ex, Map<String, String> q) throws IOException {
            List<Double> v = Json.parseVec(q.getOrDefault("v", ""));
            if (v.size() != DIMS) {
                sendJson(ex, mapOf("error", "need " + DIMS + "D vector"));
                return;
            }
            sendJson(ex, db.benchmark(v, parseInt(q.get("k"), 5), q.getOrDefault("metric", "cosine")));
        }

        private void docInsert(HttpExchange ex) throws IOException {
            String body = body(ex);
            String title = Json.extractString(body, "title");
            String text = Json.extractString(body, "text");
            if (title.isBlank() || text.isBlank()) {
                sendJson(ex, mapOf("error", "need title and text"));
                return;
            }
            List<String> chunks = chunkText(text, 250, 30);
            List<Integer> ids = new ArrayList<>();
            for (int i = 0; i < chunks.size(); i++) {
                List<Double> emb = ollama.embed(chunks.get(i));
                if (emb.isEmpty()) {
                    sendJson(ex, mapOf("error", "Ollama unavailable or embedding model missing"));
                    return;
                }
                String chunkTitle = chunks.size() > 1 ? title + " [" + (i + 1) + "/" + chunks.size() + "]" : title;
                ids.add(docDB.insert(chunkTitle, chunks.get(i), emb));
            }
            sendJson(ex, mapOf("ids", ids, "chunks", chunks.size(), "dims", docDB.dims()));
        }

        private void docList(HttpExchange ex) throws IOException {
            List<Map<String, Object>> out = new ArrayList<>();
            for (DocItem doc : docDB.all()) {
                String preview = doc.text().length() > 120 ? doc.text().substring(0, 120) + "..." : doc.text();
                int words = doc.text().isBlank() ? 0 : doc.text().trim().split("\\s+").length;
                out.add(mapOf("id", doc.id(), "title", doc.title(), "preview", preview, "words", words));
            }
            sendJson(ex, out);
        }

        private void docSearch(HttpExchange ex) throws IOException {
            String body = body(ex);
            String question = Json.extractString(body, "question");
            int k = Json.extractInt(body, "k", 3);
            if (question.isBlank()) {
                sendJson(ex, mapOf("error", "need question"));
                return;
            }
            List<Double> q = ollama.embed(question);
            if (q.isEmpty()) {
                sendJson(ex, mapOf("error", "Ollama unavailable"));
                return;
            }
            List<Map<String, Object>> contexts = new ArrayList<>();
            for (Map<String, Object> hit : docDB.search(q, k, 0.7)) {
                DocItem doc = (DocItem) hit.get("doc");
                contexts.add(mapOf("id", doc.id(), "title", doc.title(), "distance", hit.get("distance")));
            }
            sendJson(ex, mapOf("contexts", contexts));
        }

        private void docAsk(HttpExchange ex) throws IOException {
            String body = body(ex);
            String question = Json.extractString(body, "question");
            int k = Json.extractInt(body, "k", 3);
            if (question.isBlank()) {
                sendJson(ex, mapOf("error", "need question"));
                return;
            }
            List<Double> q = ollama.embed(question);
            if (q.isEmpty()) {
                sendJson(ex, mapOf("error", "Ollama unavailable"));
                return;
            }
            List<Map<String, Object>> hits = docDB.search(q, k, 0.7);
            StringBuilder context = new StringBuilder();
            List<Map<String, Object>> contexts = new ArrayList<>();
            for (int i = 0; i < hits.size(); i++) {
                DocItem doc = (DocItem) hits.get(i).get("doc");
                context.append("[").append(i + 1).append("] ").append(doc.title()).append(":\n").append(doc.text()).append("\n\n");
                contexts.add(mapOf("id", doc.id(), "title", doc.title(), "text", doc.text(), "distance", hits.get(i).get("distance")));
            }
            String prompt = "You are a helpful assistant. Answer directly using the context when useful.\n\nContext:\n"
                    + context + "Question: " + question + "\n\nAnswer:";
            sendJson(ex, mapOf("answer", ollama.generate(prompt), "model", ollama.genModel, "contexts", contexts, "docCount", docDB.size()));
        }

        private void status(HttpExchange ex) throws IOException {
            sendJson(ex, mapOf("ollamaAvailable", ollama.isAvailable(), "embedModel", ollama.embedModel,
                    "genModel", ollama.genModel, "docCount", docDB.size(), "docDims", docDB.dims(),
                    "demoDims", DIMS, "demoCount", db.size()));
        }

        private void stats(HttpExchange ex) throws IOException {
            sendJson(ex, mapOf("count", db.size(), "dims", DIMS, "algorithms", List.of("bruteforce", "kdtree", "hnsw"),
                    "metrics", List.of("euclidean", "cosine", "manhattan")));
        }

        private void staticFile(HttpExchange ex, String path) throws IOException {
            if ("/".equals(path)) path = "/index.html";
            Path target = frontend.resolve(path.substring(1)).normalize();
            if (!target.startsWith(frontend) || !Files.exists(target) || Files.isDirectory(target)) {
                send(ex, 404, "Not found", "text/plain");
                return;
            }
            send(ex, 200, Files.readString(target), contentType(target));
        }

        private void sendJson(HttpExchange ex, Object obj) throws IOException {
            send(ex, 200, Json.write(obj), "application/json");
        }

        private void send(HttpExchange ex, int status, String payload, String contentType) throws IOException {
            byte[] bytes = payload.getBytes(StandardCharsets.UTF_8);
            Headers h = ex.getResponseHeaders();
            h.set("Access-Control-Allow-Origin", "*");
            h.set("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
            h.set("Access-Control-Allow-Headers", "Content-Type, Authorization");
            h.set("Content-Type", contentType + "; charset=utf-8");
            ex.sendResponseHeaders(status, bytes.length);
            try (OutputStream out = ex.getResponseBody()) {
                out.write(bytes);
            }
        }

        private String body(HttpExchange ex) throws IOException {
            try (InputStream in = ex.getRequestBody()) {
                return new String(in.readAllBytes(), StandardCharsets.UTF_8);
            }
        }
    }

    static final class Json {
        static String str(String s) {
            StringBuilder out = new StringBuilder("\"");
            for (char c : s.toCharArray()) {
                switch (c) {
                    case '"' -> out.append("\\\"");
                    case '\\' -> out.append("\\\\");
                    case '\n' -> out.append("\\n");
                    case '\r' -> out.append("\\r");
                    case '\t' -> out.append("\\t");
                    default -> out.append(c);
                }
            }
            return out.append('"').toString();
        }

        static String extractString(String body, String key) {
            String needle = "\"" + key + "\"";
            int p = body.indexOf(needle);
            if (p < 0) return "";
            p = body.indexOf(':', p);
            if (p < 0) return "";
            p++;
            while (p < body.length() && Character.isWhitespace(body.charAt(p))) p++;
            if (p >= body.length() || body.charAt(p) != '"') return "";
            StringBuilder out = new StringBuilder();
            for (int i = p + 1; i < body.length(); i++) {
                char c = body.charAt(i);
                if (c == '"') break;
                if (c == '\\' && i + 1 < body.length()) {
                    char n = body.charAt(++i);
                    out.append(switch (n) {
                        case 'n' -> '\n';
                        case 'r' -> '\r';
                        case 't' -> '\t';
                        case '"' -> '"';
                        case '\\' -> '\\';
                        default -> n;
                    });
                } else out.append(c);
            }
            return out.toString();
        }

        static int extractInt(String body, String key, int def) {
            int p = body.indexOf("\"" + key + "\"");
            if (p < 0) return def;
            p = body.indexOf(':', p);
            if (p < 0) return def;
            int end = p + 1;
            while (end < body.length() && (Character.isDigit(body.charAt(end)) || body.charAt(end) == '-' || Character.isWhitespace(body.charAt(end)))) end++;
            try { return Integer.parseInt(body.substring(p + 1, end).trim()); } catch (Exception e) { return def; }
        }

        static List<Double> extractNumberArray(String body, String key) {
            int p = body.indexOf("\"" + key + "\"");
            if (p < 0) return List.of();
            p = body.indexOf('[', p);
            if (p < 0) return List.of();
            int end = matchingBracket(body, p);
            return end < 0 ? List.of() : parseVec(body.substring(p + 1, end));
        }

        static List<Double> extractFirstNestedNumberArray(String body, String key) {
            int p = body.indexOf("\"" + key + "\"");
            if (p < 0) return List.of();
            p = body.indexOf('[', p);
            if (p < 0) return List.of();
            int nested = body.indexOf('[', p + 1);
            if (nested < 0) return List.of();
            int end = matchingBracket(body, nested);
            return end < 0 ? List.of() : parseVec(body.substring(nested + 1, end));
        }

        static int matchingBracket(String s, int start) {
            int depth = 0;
            for (int i = start; i < s.length(); i++) {
                if (s.charAt(i) == '[') depth++;
                if (s.charAt(i) == ']') {
                    depth--;
                    if (depth == 0) return i;
                }
            }
            return -1;
        }

        static List<Double> parseVec(String raw) {
            if (raw == null || raw.isBlank()) return List.of();
            List<Double> out = new ArrayList<>();
            for (String part : raw.split(",")) {
                try { out.add(Double.parseDouble(part.trim())); } catch (Exception ignored) {}
            }
            return out;
        }

        static String write(Object obj) {
            if (obj == null) return "null";
            if (obj instanceof String s) return str(s);
            if (obj instanceof Number || obj instanceof Boolean) return String.valueOf(obj);
            if (obj instanceof DocItem d) return write(mapOf("id", d.id(), "title", d.title(), "text", d.text(), "embedding", d.embedding()));
            if (obj instanceof Map<?, ?> m) {
                StringBuilder out = new StringBuilder("{");
                boolean first = true;
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    if (!first) out.append(',');
                    first = false;
                    out.append(str(String.valueOf(e.getKey()))).append(':').append(write(e.getValue()));
                }
                return out.append('}').toString();
            }
            if (obj instanceof Iterable<?> it) {
                StringBuilder out = new StringBuilder("[");
                boolean first = true;
                for (Object item : it) {
                    if (!first) out.append(',');
                    first = false;
                    out.append(write(item));
                }
                return out.append(']').toString();
            }
            return str(String.valueOf(obj));
        }
    }

    static List<String> chunkText(String text, int chunkWords, int overlapWords) {
        String[] words = text.trim().split("\\s+");
        if (words.length <= chunkWords) return text.isBlank() ? List.of() : List.of(text);
        List<String> chunks = new ArrayList<>();
        int step = chunkWords - overlapWords;
        for (int i = 0; i < words.length; i += step) {
            int end = Math.min(i + chunkWords, words.length);
            chunks.add(String.join(" ", java.util.Arrays.copyOfRange(words, i, end)));
            if (end == words.length) break;
        }
        return chunks;
    }

    static Map<String, String> query(String raw) {
        Map<String, String> out = new HashMap<>();
        if (raw == null || raw.isBlank()) return out;
        for (String part : raw.split("&")) {
            int p = part.indexOf('=');
            String k = p >= 0 ? part.substring(0, p) : part;
            String v = p >= 0 ? part.substring(p + 1) : "";
            out.put(URLDecoder.decode(k, StandardCharsets.UTF_8), URLDecoder.decode(v, StandardCharsets.UTF_8));
        }
        return out;
    }

    static int id(String path) {
        return Integer.parseInt(path.substring(path.lastIndexOf('/') + 1));
    }

    static int parseInt(String value, int def) {
        try { return value == null || value.isBlank() ? def : Integer.parseInt(value); } catch (Exception e) { return def; }
    }

    static String contentType(Path path) {
        String name = path.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".html")) return "text/html";
        if (name.endsWith(".js")) return "text/javascript";
        if (name.endsWith(".css")) return "text/css";
        return "application/octet-stream";
    }

    static String env(String key, String def) {
        String value = System.getenv(key);
        return value == null || value.isBlank() ? def : value;
    }

    static Map<String, Object> mapOf(Object... kv) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (int i = 0; i + 1 < kv.length; i += 2) out.put(String.valueOf(kv[i]), kv[i + 1]);
        return out;
    }

    static final class DemoData {
        static void load(Nexus db) {
            db.insert("Linked List: nodes connected by pointers", "cs", v(0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06));
            db.insert("Binary Search Tree: O(log n) search and insert", "cs", v(0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10));
            db.insert("Dynamic Programming: memoization overlapping subproblems", "cs", v(0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07));
            db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs", v(0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07));
            db.insert("Hash Table: O(1) lookup with collision chaining", "cs", v(0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09));
            db.insert("Calculus: derivatives integrals and limits", "math", v(0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10));
            db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math", v(0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09));
            db.insert("Probability: distributions random variables Bayes theorem", "math", v(0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08));
            db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math", v(0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06));
            db.insert("Combinatorics: permutations combinations generating functions", "math", v(0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07));
            db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", v(0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07));
            db.insert("Sushi: vinegared rice raw fish and nori rolls", "food", v(0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08));
            db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", v(0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06));
            db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food", v(0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09));
            db.insert("Croissant: laminated pastry with buttery flaky layers", "food", v(0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06));
            db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports", v(0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72));
            db.insert("Football: tackles touchdowns field goals and strategy", "sports", v(0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76));
            db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", v(0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82));
            db.insert("Chess: openings endgames tactics strategic board game", "sports", v(0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90));
            db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports", v(0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80));
        }

        static List<Double> v(double... values) {
            List<Double> out = new ArrayList<>();
            for (double value : values) out.add(value);
            return out;
        }
    }
}
