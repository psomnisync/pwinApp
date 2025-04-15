# --- 1. Libraries ---

library(shiny)
library(httr)
library(jsonlite)
library(tm)          
library(topicmodels) 
library(proxy)       
library(textstem)    
library(purrr)
library(topicdoc)
library(stringr)

# --- 2. Helper Functions ---

'%!in%' = function(x, y)!('%in%'(x, y))

# --- 3. Shiny UI Definition ---
ui <- fluidPage(
  titlePanel("pWin Predictor"),
  
  sidebarLayout(
    sidebarPanel(
      textAreaInput("text_input0", "Company Name:", rows = 2, placeholder = "Enter Company Name."),
      textAreaInput("text_input1", "Product & Technology Description:", rows = 6, placeholder = "Enter a description of your products and technologies."),
      textAreaInput("text_input2", "Opportunity Objective:", rows = 6, placeholder = "Enter the text of the opportunity objective."),
      actionButton("submit_btn", "Calculate pWin", icon = icon("calculator")),
      hr(),
      helpText("This app calculates pWin.")
    ),
    
    mainPanel(
      h4("pWin:"),
      verbatimTextOutput("pWin_output", placeholder = TRUE) # Shows placeholder until calculation
    )
  )
)

# --- 4. Shiny Server Logic ---
server <- function(input, output, session) {
  
  # Reactive value to store the final pWin score
  pWin_result <- reactiveVal("Please enter text in all fields and click 'Calculate pWin'.")
  
  observeEvent(input$submit_btn, {
    
    showModal(modalDialog("Processing... Please wait.", footer = NULL, easyClose = FALSE))
    on.exit(removeModal(), add = TRUE)
    pWin_result("Calculating...")
    
    req(input$text_input0, input$text_input1, input$text_input2, cancelOutput = TRUE) # Ensure inputs are not empty
    if (nchar(trimws(input$text_input0)) == 0 || nchar(trimws(input$text_input1)) == 0 || nchar(trimws(input$text_input2)) == 0) {
      pWin_result("All inputs must contain text.")
      return()
    }

    user_text0 = input$text_input0
    user_text1 <- input$text_input1
    user_text2 <- input$text_input2
    
    # --- API Call ---
    api_url = "https://ipnpgbm3a6.execute-api.us-west-2.amazonaws.com/prod/query"
    token = "cFeJCO2Mow3XLYtBWGbYj9mmDUGtFiMY1zJCduZ0"

b1 = jsonlite::toJSON(list("query" = user_text2,
                           "function" = "awardees-meta",
                           "score_limit" = "medium",
                           "limit" = 300,
                           jsonlite::toJSON(list("sam_extract_code" = "A"), 
                                            auto_unbox = TRUE),
                           "filter_limit" = 300,
                           "sort_order" = "relevance"),
                      auto_unbox = TRUE)

    response = httr::POST(api_url,
                          body = b1,
                          encode = "json",
                          httr::timeout(30),
                          httr::accept_json(),
                          httr::add_headers("X-API-Key" = token))

    content_parsed = httr::content(response, "parsed", encoding = "UTF-8")
    x = data.frame(matrix(ncol = 2, nrow = 0))
    x = as.data.frame(rbind(x, c(content_parsed$body[1][[1]]$company,
                                           content_parsed$body[1][[1]]$description)))
    names(x) = c("doc_id", "text")
    for (i in 2:length(content_parsed$body)){
      x = as.data.frame(rbind(x, c(content_parsed$body[i][[1]]$company,
                                             content_parsed$body[i][[1]]$description)))
    }

    x = as.data.frame(rbind(x, c(user_text0, user_text1)))
    
    x$doc_id = toupper(x$doc_id)
    x$doc_id = gsub("[[:punct:]]* *(\\w+[&'-]\\w+)|[[:punct:]]+ *| {2,}", " \\1", x$doc_id)
    x$doc_id = trimws(x$doc_id, which = c("both", "left", "right"), whitespace = "[ \t\r\n]")
    x$doc_id = gsub("(?<=[\\s])\\s*|^\\s+|\\s+$", "", x$doc_id, perl = TRUE)
    x$doc_id = gsub("^L L C$", "LLC", x$doc_id)
    x$doc_id = gsub("^L P$", "LP", x$doc_id)
    x$doc_id = str_replace_all(x$doc_id, c("^LLC$" = "", "^INC$" = "", "^CORP$" = ""))
    x$doc_id = trimws(x$doc_id, which = c("both", "left", "right"), whitespace = "[ \t\r\n]")
    x$doc_id = gsub("(?<=[\\s])\\s*|^\\s+|\\s+$", "", x$doc_id, perl = TRUE)
    cName = x$doc_id[nrow(x)]
    x = x %>% group_by(doc_id) %>% summarize(text = paste(text, collapse = " "))

    proj.corpus = Corpus(VectorSource(x$text))
    proj.corpus = tm_map(proj.corpus, content_transformer(tolower))
    proj.corpus = tm_map(proj.corpus, removePunctuation)
    proj.corpus = tm_map(proj.corpus, removeNumbers)
    proj.corpus = tm_map(proj.corpus, removeWords, stopwords("en"))
    proj.corpus = tm_map(proj.corpus, stripWhitespace)
    proj.corpus = tm_map(proj.corpus, stemDocument, language = "en")
    proj.dtm = DocumentTermMatrix(proj.corpus)
    sel_idx = apply(proj.dtm, 1, sum)
    proj.dtm = proj.dtm[sel_idx > 0, ]
    x = x[sel_idx > 0,]
    if(any(slam::row_sums(proj.dtm) == 0)) {
      warning("Removing empty rows from dtm...")
      proj.dtm = proj.dtm[slam::row_sums(proj.dtm) > 0, ]
    }

    ##### Define Hyperparameter Grid #####

    k_vector = c(8:10)
    alpha_vector = c(50/k_vector, 0.5, 0.1, 1/k_vector)
    delta_vector = c(0.1, 0.05, 0.01)

    # Create a data frame to store results
    tuning_results = data.frame(k = integer(),
                                alpha = numeric(),
                                delta = numeric(),
                                mean_coherence = numeric(),
                                stringsAsFactors = FALSE)
    ###### Tuning Loop #####

    # print(paste("Starting hyperparameter optimization for Project",
    #             toupper(projNames[i]), "at:", Sys.time()))
    total_combinations = length(k_vector) * length(alpha_vector) * length(delta_vector)
    current_combination = 0

    for (k_val in k_vector) {
      for (alpha_val in alpha_vector) {
        for (delta_val in delta_vector) {
          current_combination = current_combination + 1
          print(paste("Running combination", current_combination, "/", total_combinations,
                      "- k:", k_val, "alpha:", round(alpha_val,3), "delta:", delta_val))
          K = k_val
          tm = LDA(proj.dtm, K, method = "Gibbs",
                   control = list(seed = 20250407,
                                  iter = 500,
                                  alpha = alpha_val,
                                  delta = delta_val,
                                  verbose = 250))
          tmResult = posterior(tm)
          coherence_scores = topicdoc::topic_coherence(topic_model = tm,
                                                       dtm_data = proj.dtm)
          mean_coherence = mean(coherence_scores, na.rm = TRUE)

          # Store results
          tuning_results = rbind(tuning_results,
                                 data.frame(k = k_val,
                                            alpha = alpha_val,
                                            delta = delta_val,
                                            mean_coherence = mean_coherence))
        } # end delta loop
        # beep()
      } # end alpha loop
      # beep()
    } # end k loop

    # print(paste("Finished hyperparameter optimization for Project ", toupper(projNames[i]),
    #             "at:", Sys.time()))
    # beep()

    ##### Optimal Parameter Selection #####

    best_params = tuning_results %>%
      filter(!is.na(mean_coherence)) %>%
      arrange(desc(mean_coherence))

    # print("--- Tuning Results ---")
    # print(tuning_results)

    # assign(paste0(projNames[i], ".tuning_results"), tuning_results)
    # assign(paste0(projNames[i], ".best_params"), best_params)

    # print("--- Best Parameters based on Mean Coherence ---")

    if (nrow(best_params) > 0) {
      # print(head(best_params, 5))
      optimal_k = best_params$k[1]
      optimal_alpha = best_params$alpha[1]
      optimal_delta = best_params$delta[1]
      # print(paste("Optimal k:", optimal_k))
      # print(paste("Optimal alpha:", optimal_alpha))
      # print(paste("Optimal delta:", optimal_delta))
      # assign("proj.optimal_k", optimal_k)
      # assign("proj.optimal_alpha", optimal_alpha)
      # assign("proj.optimal_delta", optimal_delta)
    } else {
      print("No successful runs completed. Check warnings and errors.")
      optimal_k = NA
      optimal_alpha = NA
      optimal_delta = NA
    }
    # 
    ##### Final Topic Model #####

    final_control_gibbs = list(
      alpha = optimal_alpha,
      delta = optimal_delta,
      seed = 20250407,
      burnin = 1000,
      iter = 2500,
      thin = 100,
      verbose = 250
    )
    proj.tm = LDA(proj.dtm,
                  k = optimal_k,
                  method = "Gibbs",
                  control = final_control_gibbs)

    proj.tmResult = posterior(proj.tm)
    proj.beta = proj.tmResult$terms
    proj.theta = proj.tmResult$topics
    terms(proj.tm, 10)
    proj.top3termsPerTopic = terms(proj.tm, 3)
    proj.top3termsPerTopic
    proj.topNames = apply(lda::top.topic.words(proj.beta, 3, by.score = T), 2, paste, collapse = ", ")
    # proj.json = createJSON(
    #   phi = proj.beta,
    #   theta = proj.theta,
    #   doc.length = slam::row_sums(proj.dtm),
    #   vocab = colnames(proj.dtm),
    #   term.frequency = slam::col_sums(proj.dtm),
    #   mds.method = svd_tsne,
    #   plot.opts = list(xlab = "", ylab ="")
    # )
    #
    # assign(paste0(as.character(projNames[i]), ".tm"), proj.tm)
    # assign(paste0(as.character(projNames[i]), ".tmResult"), proj.tmResult)
    # assign(paste0(as.character(projNames[i]), ".beta"), proj.beta)
    # assign(paste0(as.character(projNames[i]), ".theta"), proj.theta)
    # assign(paste0(as.character(projNames[i]), ".t3tpt"), proj.top3termsPerTopic)
    # assign(paste0(as.character(projNames[i]), ".topNames"), proj.topNames)
    # assign(paste0(as.character(projNames[i]), ".LDAvisJSON"), proj.json)

    proj.topNums = optimal_k
    proj.sol = user_text2
    proj.theta2 = as.data.frame(proj.theta)
    proj.theta2$doc_id = NA
    proj.theta2 = proj.theta2[,c(ncol(proj.theta2),1:proj.topNums)]
    colnames(proj.theta2)[2:ncol(proj.theta2)] = sapply(c(1:proj.topNums),
                                                        function(j) paste0("topic", j))
    proj.theta2$doc_id = x$doc_id
    proj.ogTerms = Terms(proj.dtm)

    proj.sol.corpus = VCorpus(VectorSource(proj.sol))
    proj.sol.corpus = tm_map(proj.sol.corpus, content_transformer(tolower))
    proj.sol.corpus = tm_map(proj.sol.corpus, removePunctuation)
    proj.sol.corpus = tm_map(proj.sol.corpus, removeNumbers)
    proj.sol.corpus = tm_map(proj.sol.corpus, removeWords, stopwords("en"))
    proj.sol.corpus = tm_map(proj.sol.corpus, stripWhitespace)
    proj.sol.corpus = tm_map(proj.sol.corpus, stemDocument, language = "en")

    proj.sol.dtm = DocumentTermMatrix(proj.sol.corpus,
                                      control = list(dictionary = proj.ogTerms))
    proj.sol.topics.pred = posterior(proj.tm, newdata = proj.sol.dtm)

    proj.theta3 = proj.theta2
    proj.theta3[nrow(proj.theta3) + 1, 1:ncol(proj.theta3)] = c("sol", proj.sol.topics.pred$topics)

    cosine = sapply(1:nrow(proj.theta3),
                    function(x) simil(list(as.numeric(proj.theta3[nrow(proj.theta3), 2:ncol(proj.theta3)]),
                                           as.numeric(proj.theta3[x, 2:ncol(proj.theta3)])),
                                      method = "cosine"))
    Hellinger = sapply(1:nrow(proj.theta3),
                       function(x) simil(list(as.numeric(proj.theta3[nrow(proj.theta3), 2:ncol(proj.theta3)]),
                                              as.numeric(proj.theta3[x, 2:ncol(proj.theta3)])),
                                         method = "Hellinger"))
    Kullback = sapply(1:nrow(proj.theta3),
                      function(x) simil(list(as.numeric(proj.theta3[nrow(proj.theta3), 2:ncol(proj.theta3)]),
                                             as.numeric(proj.theta3[x, 2:ncol(proj.theta3)])),
                                        method = "Kullback"))
    proj.theta3[,c((ncol(proj.theta3) + 1):(ncol(proj.theta3) + 3))] = as.data.frame(cbind(cosine,
                                                                                           Hellinger,
                                                                                           Kullback))
    proj.theta3$pWin = ((proj.theta3$cosine^2 + proj.theta3$Hellinger^2 + proj.theta3$Kullback^2)/3)^0.5
    proj.theta3[2:ncol(proj.theta3)] = sapply(proj.theta3[2:ncol(proj.theta3)], as.numeric)

    finalPWinDF = proj.theta3[,c(1,ncol(proj.theta3))]
    finalPWinDF = finalPWinDF[order(-finalPWinDF$pWin),]
    row.names(finalPWinDF) = NULL

    pWin_result(round(finalPWinDF[which(finalPWinDF$doc_id == cName), 2]*100, 2))
    
    # pWin_result(optimal_alpha)
    
  }) # end observeEvent
  
  # --- Render Final Output ---
  output$pWin_output <- renderText({
    pWin_result() 
  })
  
} # end server

# --- 5. Run the Application ---
shinyApp(ui = ui, server = server)