install.packages(c("tidyverse", "tm", "SnowballC", "proxy", "cluster", "syuzhet", "igraph", "ggraph", "tidygraph", "ggplot2"))

library(tidyverse)
library(tm)
library(SnowballC)
library(proxy)  
library(cluster)
library(syuzhet)
library(igraph)
library(ggraph)
library(tidygraph)
library(ggplot2)

# Load the dataset
corpus_df <- read.csv("data/corpus.csv", stringsAsFactors = FALSE)

# Check structure
str(corpus_df)
head(corpus_df)

table(corpus_df$genre)

# Check word counts
corpus_df$word_count <- sapply(strsplit(corpus_df$text, "\\s+"), length)

summary(corpus_df$word_count)

# QUESTION 3: Text Processing and Document-Term Matrix

# Create text corpus from the text column
text_corpus <- VCorpus(VectorSource(corpus_df$text))

# Basic text cleaning
text_corpus <- tm_map(text_corpus, content_transformer(tolower))
text_corpus <- tm_map(text_corpus, removePunctuation)
text_corpus <- tm_map(text_corpus, removeNumbers)
text_corpus <- tm_map(text_corpus, removeWords, stopwords("english"))
text_corpus <- tm_map(text_corpus, stripWhitespace)

# stemming
text_corpus <- tm_map(text_corpus, stemDocument)

# Create initial DTM
dtm <- DocumentTermMatrix(text_corpus)

dtm

# Trial-and-error sparse term removal
# Increase/decrease this value unitl around 25 terms remain
dtm_sparse <- removeSparseTerms(dtm, 0.20)

dtm_sparse

# Convert dtm to matrix
dtm_matrix <- as.matrix(dtm_sparse)

# Add document IDs as row names
rownames(dtm_matrix) <- corpus_df$doc_id

# View final tokens
colnames(dtm_matrix)

write.csv(dtm_matrix, "data/dtm_matrix.csv", row.names = TRUE)

# Save token frquency table
token_freq <- colSums(dtm_matrix)
token_freq_df <- data.frame(
    token = names(token_freq),
    frequency = as.numeric(token_freq)
)  %>%
    arrange(desc(frequency))

write.csv(token_freq_df, "data/token_freq.csv", row.names = FALSE)

# Print number of final tokens
cat("Number of final tokens:", ncol(dtm_matrix), "\n")

# ============================================================
# QUESTION 4: Hierarchical Clustering of Documents
# ============================================================

library(dendextend)

# Cosine distance
cos_dist <- proxy::dist(dtm_matrix, method = "cosine")

# Hierarchical clustering
hc <- hclust(cos_dist, method = "ward.D2")

# Convert to dendrogram
dend <- as.dendrogram(hc)

# Genre colours
genre_colours <- c(
  "Film Review" = "#E69F00",
  "News" = "#0072B2",
  "Sports" = "#009E73"
)

label_genres <- corpus_df$genre[match(labels(dend), corpus_df$doc_id)]

labels_colors(dend) <- genre_colours[label_genres]
labels_cex(dend) <- 0.9

# Save HD dendrogram
png(
  filename = "figures/dendrogram_hd.png",
  width = 2400,
  height = 1600,
  res = 250
)

par(mar = c(8, 5, 5, 2))

plot(
  dend,
  main = "Hierarchical Clustering of Documents Using Cosine Distance",
  ylab = "Height",
  xlab = "",
  sub = "Labels are coloured by actual genre"
)

rect.hclust(hc, k = 3, border = "red")

legend(
  "topright",
  legend = names(genre_colours),
  fill = genre_colours,
  border = NA,
  cex = 0.9,
  title = "Genre"
)

dev.off()

# Cluster assignment
cluster_assignments <- cutree(hc, k = 3)

cluster_results <- data.frame(
  doc_id = corpus_df$doc_id,
  title = corpus_df$title,
  genre = corpus_df$genre,
  cluster = cluster_assignments
)

write.csv(cluster_results, "data/cluster_results.csv", row.names = FALSE)

# Genre-cluster table
cluster_table <- table(cluster_results$genre, cluster_results$cluster)
print(cluster_table)

write.csv(
  as.data.frame.matrix(cluster_table),
  "data/cluster_genre_table.csv"
)

# Accuracy by majority genre
cluster_accuracy <- sum(apply(cluster_table, 2, max)) / sum(cluster_table)

cat("Clustering Accuracy:", round(cluster_accuracy * 100, 2), "%\n")

accuracy_df <- data.frame(
  metric = "Clustering accuracy by majority genre",
  value = round(cluster_accuracy * 100, 2)
)

write.csv(
  accuracy_df,
  "data/clustering_accuracy.csv",
  row.names = FALSE
)

install.packages("ggpubr")

# ============================================================
# QUESTION 5: SENTIMENT ANALYSIS
# ============================================================

library(syuzhet)
library(ggplot2)

# Get sentiment score for each document
sentiment_scores <- get_sentiment(
  corpus_df$text,
  method = "syuzhet"
)

# Create results table
sentiment_df <- data.frame(
  doc_id = corpus_df$doc_id,
  genre = corpus_df$genre,
  sentiment = sentiment_scores
)

print(sentiment_df)

# Save results
write.csv(
  sentiment_df,
  "data/sentiment_results.csv",
  row.names = FALSE
)

# Genre summary
genre_sentiment <- aggregate(
  sentiment ~ genre,
  data = sentiment_df,
  FUN = mean
)

print(genre_sentiment)

write.csv(
  genre_sentiment,
  "data/genre_sentiment.csv",
  row.names = FALSE
)

# Sentiment Boxplot
p1 <- ggplot(
    sentiment_df,
    aes(
        x = genre,
        y = sentiment,
        fill = genre
    )
) +
    geom_boxplot(alpha = 0.8) +
    geom_jitter(width = 0.1) +
    labs(
        title = "Sentiment Distribution by Genre",
        x = "Genre",
        y = "Sentiment Score"
    ) +
    theme_minimal() +
    theme(
        legend.position = "none",
        plot.title = element_text(face = "bold")
    )

ggsave(
    "figures/sentiment_boxplot.png",
    p1,
    width = 8,
    height = 5
)

# ============================================================
# ANOVA TEST
# ============================================================

anova_result <- aov(
  sentiment ~ genre,
  data = sentiment_df
)

summary(anova_result)

capture.output(
  summary(anova_result),
  file = "report/anova_results.txt"
)

library(dplyr)

sentiment_summary <- sentiment_df %>%
  group_by(genre) %>%
  summarise(
    mean_sentiment = mean(sentiment),
    sd_sentiment = sd(sentiment),
    min_sentiment = min(sentiment),
    max_sentiment = max(sentiment)
  )

print(sentiment_summary)

write.csv(
  sentiment_summary,
  "data/sentiment_summary.csv",
  row.names = FALSE
)

# ============================================================
# QUESTION 6: DOCUMENT NETWORK USING TF-IDF SIMILARITY
# ============================================================

library(tm)
library(proxy)
library(igraph)
library(ggraph)
library(tidygraph)
library(dplyr)
library(ggplot2)

# Create a TF-IDF weighted DTM from the cleaned corpus
dtm_tfidf <- DocumentTermMatrix(
  text_corpus,
  control = list(weighting = weightTfIdf)
)

# Remove very sparse terms, but keep more detail than Q3
dtm_tfidf_sparse <- removeSparseTerms(dtm_tfidf, 0.80)

tfidf_matrix <- as.matrix(dtm_tfidf_sparse)
rownames(tfidf_matrix) <- corpus_df$doc_id

cat("TF-IDF network tokens:", ncol(tfidf_matrix), "\n")

# Cosine similarity between documents
cosine_distance <- proxy::dist(tfidf_matrix, method = "cosine")
cosine_similarity <- 1 - as.matrix(cosine_distance)

diag(cosine_similarity) <- 0

# Convert similarity matrix to edge list
doc_edges <- as.data.frame(as.table(cosine_similarity))
colnames(doc_edges) <- c("from", "to", "similarity")

doc_edges <- doc_edges %>%
  mutate(
    from = as.character(from),
    to = as.character(to),
    similarity = as.numeric(similarity)
  ) %>%
  filter(from < to)

# Keep only strongest similarities
# Adjust this threshold if graph is too dense/sparse
similarity_threshold <- quantile(doc_edges$similarity, 0.75)

doc_edges <- doc_edges %>%
  filter(similarity >= similarity_threshold)

summary(doc_edges$similarity)

# Node table
doc_nodes <- data.frame(
  name = corpus_df$doc_id,
  genre = corpus_df$genre,
  sentiment = sentiment_df$sentiment
)

# Create graph
doc_graph <- graph_from_data_frame(
  d = doc_edges,
  vertices = doc_nodes,
  directed = FALSE
)

# Community detection
communities <- cluster_louvain(doc_graph, weights = E(doc_graph)$similarity)
V(doc_graph)$community <- communities$membership

# Centrality
V(doc_graph)$degree <- degree(doc_graph)
V(doc_graph)$strength <- strength(doc_graph, weights = E(doc_graph)$similarity)
V(doc_graph)$betweenness <- betweenness(doc_graph, weights = 1 / E(doc_graph)$similarity)

doc_centrality <- data.frame(
  doc_id = V(doc_graph)$name,
  genre = V(doc_graph)$genre,
  community = V(doc_graph)$community,
  degree = V(doc_graph)$degree,
  strength = round(V(doc_graph)$strength, 3),
  betweenness = round(V(doc_graph)$betweenness, 3),
  sentiment = V(doc_graph)$sentiment
) %>%
  arrange(desc(strength), desc(degree))

write.csv(
  doc_centrality,
  "data/document_centrality.csv",
  row.names = FALSE
)

# ============================================================
# Improved document network plot
# ============================================================

set.seed(123)

p_doc_network <- ggraph(doc_graph, layout = "kk") +
  geom_edge_link(
    aes(width = similarity),
    colour = "grey65",
    alpha = 0.45
  ) +
  geom_node_point(
    aes(size = strength, colour = genre),
    alpha = 0.95
  ) +
  geom_node_text(
    aes(label = name),
    repel = TRUE,
    size = 4
  ) +
  scale_edge_width(
    range = c(0.3, 2.5),
    name = "TF-IDF cosine similarity"
  ) +
  scale_size_continuous(
    range = c(4, 10),
    name = "Strength centrality"
  ) +
  labs(
    title = "Document Network Based on TF-IDF Similarity",
    subtitle = "Nodes are documents; edges show strongest cosine similarities between documents",
    colour = "Genre"
  ) +
  theme_graph(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom",
    guides(size = "none")
  )

ggsave(
  "figures/document_network_hd.png",
  p_doc_network,
  width = 10,
  height = 7,
  dpi = 300
)

# ============================================================
# Supporting centrality bar chart
# ============================================================

p_doc_strength <- ggplot(
  doc_centrality,
  aes(x = reorder(doc_id, strength), y = strength, fill = genre)
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Document Strength Centrality",
    subtitle = "Higher strength means stronger similarity to other documents",
    x = "Document",
    y = "Strength centrality",
    fill = "Genre"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11)
  )

ggsave(
  "figures/document_strength.png",
  p_doc_strength,
  width = 9,
  height = 7,
  dpi = 300
)

print(doc_centrality)

# ============================================================
# QUESTION 7: TOKEN NETWORK USING TF-IDF CO-OCCURRENCE
# ============================================================

library(igraph)
library(ggraph)
library(tidygraph)
library(dplyr)
library(ggplot2)

# Use the TF-IDF matrix from Q6
token_matrix <- tfidf_matrix

# Token-token co-occurrence / similarity matrix
token_co <- t(token_matrix) %*% token_matrix

diag(token_co) <- 0

# Convert to edge list
token_edges <- as.data.frame(as.table(token_co))
colnames(token_edges) <- c("from", "to", "weight")

token_edges <- token_edges %>%
  mutate(
    from = as.character(from),
    to = as.character(to),
    weight = as.numeric(weight)
  ) %>%
  filter(from < to) %>%
  filter(weight > 0)

# Keep strongest token relationships
token_threshold <- quantile(token_edges$weight, 0.95)

token_edges <- token_edges %>%
  filter(weight >= token_threshold)

# Create token graph
token_graph <- graph_from_data_frame(
  d = token_edges,
  directed = FALSE
)

# Community detection
token_communities <- cluster_louvain(
  token_graph,
  weights = E(token_graph)$weight
)

V(token_graph)$community <- token_communities$membership

# Centrality
V(token_graph)$degree <- degree(token_graph)
V(token_graph)$strength <- strength(
  token_graph,
  weights = E(token_graph)$weight
)

V(token_graph)$betweenness <- betweenness(
  token_graph,
  weights = 1 / E(token_graph)$weight
)

token_centrality <- data.frame(
  token = V(token_graph)$name,
  community = V(token_graph)$community,
  degree = V(token_graph)$degree,
  strength = round(V(token_graph)$strength, 3),
  betweenness = round(V(token_graph)$betweenness, 3)
) %>%
  arrange(desc(strength), desc(degree))

write.csv(
  token_centrality,
  "data/token_centrality.csv",
  row.names = FALSE
)

print(head(token_centrality, 20))

# ============================================================
# TOKEN NETWORK PLOT
# ============================================================

set.seed(123)

p_token_network <- ggraph(token_graph, layout = "kk") +
  geom_edge_link(
    aes(width = weight),
    colour = "grey70",
    alpha = 0.45
  ) +
  geom_node_point(
    aes(size = strength, colour = factor(community)),
    alpha = 0.95
  ) +
  geom_node_text(
    aes(label = name),
    repel = TRUE,
    size = 4
  ) +
  scale_edge_width(
    range = c(0.2, 2.5),
    name = "Co-occurrence strength"
  ) +
  scale_size_continuous(
    range = c(4, 10),
    name = "Strength centrality"
  ) +
  labs(
    title = "Token Network Based on TF-IDF Co-occurrence",
    subtitle = "Nodes are tokens; edges show strongest relationships between words",
    colour = "Community"
  ) +
  theme_graph(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom"
  )

ggsave(
  "figures/token_network.png",
  p_token_network,
  width = 10,
  height = 8,
  dpi = 300
)

# ============================================================
# TOP TOKEN CENTRALITY BAR CHART
# ============================================================

top_tokens <- token_centrality %>%
  slice_head(n = 15)

p_top_tokens <- ggplot(
  top_tokens,
  aes(x = reorder(token, strength), y = strength)
) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 15 Tokens by Strength Centrality",
    subtitle = "Higher strength means stronger connection to other important tokens",
    x = "Token",
    y = "Strength centrality"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", size = 16),
    plot.subtitle = element_text(size = 11)
  )

ggsave(
  "figures/top_tokens_centrality.png",
  p_top_tokens,
  width = 8,
  height = 6,
  dpi = 300
)

# ============================================================
# QUESTION 8: BIPARTITE NETWORK OF DOCUMENTS AND TOKENS
# ============================================================

library(igraph)
library(ggraph)
library(dplyr)
library(ggplot2)

# Use TF-IDF matrix from Q6
bipartite_matrix <- tfidf_matrix

# Keep only top tokens by centrality from Q7 to avoid messy graph
top_bipartite_tokens <- token_centrality %>%
  slice_head(n = 20) %>%
  pull(token)

bipartite_matrix <- bipartite_matrix[, colnames(bipartite_matrix) %in% top_bipartite_tokens]

# Convert matrix to edge list
bipartite_edges <- as.data.frame(as.table(bipartite_matrix))

colnames(bipartite_edges) <- c("document", "token", "weight")

bipartite_edges <- bipartite_edges %>%
  mutate(
    document = as.character(document),
    token = as.character(token),
    weight = as.numeric(weight)
  ) %>%
  filter(weight > 0)

# Create node table
document_nodes <- data.frame(
  name = corpus_df$doc_id,
  type = "Document",
  genre = corpus_df$genre
)

token_nodes <- data.frame(
  name = top_bipartite_tokens,
  type = "Token",
  genre = "Token"
)

bipartite_nodes <- bind_rows(document_nodes, token_nodes)

# Create graph
bipartite_graph <- graph_from_data_frame(
  d = bipartite_edges,
  vertices = bipartite_nodes,
  directed = FALSE
)

# Bipartite type: TRUE for tokens, FALSE for documents
V(bipartite_graph)$bipartite_type <- V(bipartite_graph)$type == "Token"

# Centrality
V(bipartite_graph)$degree <- degree(bipartite_graph)
V(bipartite_graph)$strength <- strength(
  bipartite_graph,
  weights = E(bipartite_graph)$weight
)

bipartite_centrality <- data.frame(
  node = V(bipartite_graph)$name,
  type = V(bipartite_graph)$type,
  genre = V(bipartite_graph)$genre,
  degree = V(bipartite_graph)$degree,
  strength = round(V(bipartite_graph)$strength, 3)
) %>%
  arrange(desc(strength))

write.csv(
  bipartite_centrality,
  "data/bipartite_centrality.csv",
  row.names = FALSE
)

print(head(bipartite_centrality, 20))

# ============================================================
# BIPARTITE NETWORK PLOT
# ============================================================

set.seed(123)

p_bipartite <- ggraph(
  bipartite_graph,
  layout = "stress"
) +
  geom_edge_link(
    aes(width = weight),
    colour = "grey75",
    alpha = 0.35
  ) +
  geom_node_point(
    aes(
      shape = type,
      colour = genre,
      size = strength
    ),
    alpha = 0.95
  ) +
  geom_node_text(
    aes(label = name),
    repel = TRUE,
    size = 3.5
  ) +
  scale_edge_width(
    range = c(0.2, 2.2),
    name = "TF-IDF weight"
  ) +
  scale_size_continuous(
    range = c(3, 9),
    name = "Strength"
  ) +
  labs(
    title = "Bipartite Network of Documents and Tokens",
    subtitle = "Documents are connected to their most important tokens",
    colour = "Genre",
    shape = "Node type"
  ) +
  theme_graph(base_family = "sans") +
  theme(
    plot.title = element_text(face = "bold", size = 18),
    plot.subtitle = element_text(size = 12),
    legend.position = "bottom"
  )

ggsave(
  "figures/bipartite_network.png",
  p_bipartite,
  width = 12,
  height = 8,
  dpi = 300
)
