#!/usr/bin/env -S bb -m eval
   (ns eval
     (:require [babashka.fs :as fs]
               [babashka.process :refer [shell]]
               [cheshire.core :as json]
               [cli-matic.core :as cli]
               [clojure.java.io :as io]
               [clojure.math :as math]
               [clojure.string :as str]
               [flatland.ordered.map :refer [ordered-map]]
               [taoensso.timbre :as log]
               [camel-snake-kebab.core :as csk]
               [globus.core :as glob]
               [kixi.stats.test :as t])
     (:import [clojure.lang ExceptionInfo]))

(defn normalize-colname
  [colname]
  (-> colname
      name
      csk/->camelCase
      (str/replace #"0\.(\d+)"
                   (fn [[_ decs]] (apply str "p" (map #(char (+ 17 (int %))) decs))))
      (str/replace #"\d+" #(apply str (repeat (parse-long %) \I)))))

;; Config

(def default-ds {:run.log "False"
                 :run.num_inits 10
                 :run.num_splits 10})
(def big-default-ds (merge default-ds
                           {:model.sparse_propagation "True"}))
(def datasets
  {"CoraML" (assoc default-ds ::colname "CoraML" ::num-classes 7)
   "CiteSeerFull" (assoc default-ds ::colname "CiteSeer" ::inline-name "CiteSeer" ::num-classes 6)
   "AmazonPhotos" (assoc big-default-ds ::colname "Photos" ::inline-name "Amazon\\\\Photos" ::num-classes 8)
   "AmazonComputers" (assoc big-default-ds ::colname "Computers" ::inline-name "Amazon\\\\Computers" ::num-classes 10)
   "PubMedFull" (assoc big-default-ds ::colname "PubMed" ::inline-name "PubMed" ::num-classes 3)
   "ogbn-arxiv" (merge big-default-ds
                       {::colname "Arxiv"
                        ::inline-name "OGBN\\\\Arxiv"
                        ::num-classes 40
                        :data.split "public"
                        :run.num_splits 1
                        :run.reduced_training_metrics "True"
                        :training.eval_every 10
                        :training.stopping_patience 5
                        :run.log "True"})})

(defn create-model-dict
  []
  (let [base-models
        {"mlp" {::name "mlp"
                ::inline-name "MLP"}
         "appnp" {::name "appnp"
                  ::inline-name "APPNP"
                  ::ignored-metrics [:ood_detection_aleatoric_entropy_auroc]}
         "ggp" {::name "ggp"
                ::inline-name "GGP"
                :run.num_inits 1}
         "matern_ggp" {::name "matern_ggp"
                       ::colname "maternGGP"
                       ::inline-name "Matern-GGP"
                       ::ignored-metrics [:ood_detection_aleatoric_entropy_auroc]
                       :run.num_inits 1}
         "gdk" {::name "gdk" ::inline-name "GKDE"}
         "gcn" {::name "gcn" ::inline-name "GCN"}
         "gat" {::name "gcn"
                ::inline-name "GAT"
                :model.model_name "GAT"
                :model.heads_conv1 1
                :model.heads_conv2 1
                :model.negative_slope 0.2
                :model.coefficient_dropout_prob 0}
         "gpn_rw" {::name "gpn_16"
                   ::colname "gpnRW"
                   ::inline-name "GPN (rw)"
                   :model.adj_normalization "rw"}
         "gpn_sum" {::name "gpn_16"
                    ::colname "gpnSum"
                    ::inline-name "GPN (sum)"
                    :model.adj_normalization "sum"}
         "gpn_lop" {::name "gpn_16"
                    ::inline-name "LOP-GPN"
                    ::colname "gpnLOP"
                    :model.model_name "GPN_LOP"
                    :model.sparse_x_prune_threshold 0.01}
         "cuq_appnp" {::name "gpn_16"
                      ::colname "cuqPPR"
                      ::inline-name "CUQ-PPR"
                      :model.model_name "CUQ_GNN"
                      :model.convolution_name "appnp"}
         "cuq_gcn" {::name "gpn_16"
                    ::colname "cuqGCN"
                    ::inline-name "CUQ-GCN"
                    :model.model_name "CUQ_GNN"
                    :model.convolution_name "gcn"}
         "cuq_gat" {::name "gpn_16"
                    ::colname "cuqGAT"
                    ::inline-name "CUQ-GAT"
                    :model.model_name "CUQ_GNN"
                    :model.convolution_name "gat"
                    :model.heads_conv1 1
                    :model.heads_conv2 1
                    :model.negative_slope 0.2
                    :model.coefficient_dropout_prob 0}
         "memory_quant" {::name "memory_quant"
                         ::colname "memoryQuant"
                         :run.num_inits 1}
         "ego_net" {::name "ego_net"
                    ::inline-name "EgoNet"
                    ::colname "egoNet"
                    :run.num_inits 1
                    :run.reduced_quantification "True"}}
        qgnn-models
        {"qgnn_appnp" {::name "qgnn"
                       ::colname "qgnnPPR"
                       ::inline-name "QGNN-PPR"
                       :model.convolution_name "appnp"}
         "qgnn_gcn" {::name "qgnn"
                     ::colname "qgnnGCN"
                     ::inline-name "QGNN-GCN"
                     :model.convolution_name "gcn"}
         "qgnn_gat" {::name "qgnn"
                     ::colname "qgnnGAT"
                     ::inline-name "QGNN-GAT"
                     :model.convolution_name "gat"
                     :model.heads_conv1 1
                     :model.heads_conv2 1
                     :model.negative_slope 0.2
                     :model.coefficient_dropout_prob 0}}
        tuplet-aggregation ["nn" "mean"]
        qgnn-sampling ["random" "zipf"]
        tuplet-size-factor [#_nil 1 2 4 8 16 32]
        qgnn-models (apply merge
                           (for [a tuplet-aggregation
                                 s qgnn-sampling
                                 f tuplet-size-factor]
                             (into {}
                                   (map (fn [[k v]]
                                          [(str k "_" a "_" s (when f (str "_" f)))
                                           (assoc v
                                                  :data.quantification_train_tuplet_strategy s
                                                  :model.quantification_tuplet_aggregation a
                                                  ::tuplet-size-factor f)]))
                                   qgnn-models)))]
    (merge base-models qgnn-models)))

(def models (create-model-dict))

(def default-al-setting {::name "classification" ; reuse classification settings
                         :run.experiment_name "active_learning"
                         :run.ex_type "active_learning"
                         :run.job "train"
                         :training.al_retrain "False"
                         :training.al_initial_train_size 0.1
                         :training.al_final_train_size 1.0
                         :training.al_num_rounds 45
                         :training.al_complete_experiment_name "classification"
                         :run.num_inits 3
                         :run.per_init_variance "True"})

(def settings
  {"classification" {}
   "ood_loc" {::colname "oodLoc"}
   "ood_features_normal" {::name "ood_features"
                          ::colname "oodNormal"
                          ::depends-on "classification"
                          :data.ood_perturbation_type "normal"
                          :run.experiment_name "ood_features_normal"}
   "ood_features_ber" {::name "ood_features"
                       ::colname "oodBer"
                       ::depends-on "classification"
                       :data.ood_perturbation_type "bernoulli_0.5"
                       :run.experiment_name "ood_features_ber"}
   "active_learning" default-al-setting
   "active_learning_total" (merge default-al-setting
                                  {:training.al_criterion "sample_confidence_total"})
   "active_learning_aleatoric" (merge default-al-setting
                                      {:training.al_criterion "sample_confidence_aleatoric"})
   "active_learning_epistemic" (merge default-al-setting
                                      {:training.al_criterion "sample_confidence_epistemic"})
   "active_learning_total_entropy" (merge default-al-setting
                                          {:training.al_criterion "sample_confidence_total_entropy"})
   "active_learning_aleatoric_entropy" (merge default-al-setting
                                              {:training.al_criterion "sample_confidence_aleatoric_entropy"})
   "active_learning_epistemic_entropy_diff" (merge default-al-setting
                                                   {:training.al_criterion "sample_confidence_epistemic_entropy_diff"})
   "active_learning_epistemic_entropy" (merge default-al-setting
                                              {:training.al_criterion "sample_confidence_epistemic_entropy"})
   "skewed_quantification_zipf" {::name "classification"
                                 ::depends-on "classification"
                                 :data.quantification_test_skew "True"
                                 :data.quantification_test_skew_strategy "zipf"
                                 :data.quantification_test_skew_repeats 10
                                 :data.quantification_test_skew_tuplet_size 100}
   "skewed_quantification_neighbor" {::name "classification"
                                     ::depends-on "classification"
                                     :data.quantification_test_skew "True"
                                     :data.quantification_test_skew_strategy "neighbor"
                                     :data.quantification_test_skew_repeats 10
                                     :data.quantification_test_skew_tuplet_size 100
                                     :data.quantification_test_skew_depth_limit 5
                                     :data.quantification_test_skew_noise 0}
   "skewed_quantification_ppr" {::name "classification"
                                ::depends-on "classification"
                                :data.quantification_test_skew "True"
                                :data.quantification_test_skew_strategy "ppr"
                                :data.quantification_test_skew_repeats 10
                                :data.quantification_test_skew_tuplet_size 100
                                :data.quantification_test_skew_depth_limit 10
                                :data.quantification_test_skew_noise 0}})

(declare transform-matern-ggp-results)

(def combination-overrides
  {{::model "gpn_lop" ::dataset "PubMedFull"}
   {:run.log "True"}
   {::model "gpn_lop" ::dataset "ogbn-arxiv"}
   {:model.sparse_x_prune_threshold 0.01
    :run.num_inits 2}
   {::model "gdk" ::dataset "ogbn-arxiv"}
   {:model.gdk_cutoff 2}
   {::model "matern_ggp"}
   {::result-transform #'transform-matern-ggp-results}
   {::model "matern_ggp" ::dataset "ogbn-arxiv"}
   {::skip true}
   {::dataset "PubMedFull" ::setting "skewed_quantification_zipf"}
   {:data.quantification_test_skew_tuplet_size 300}
   {::dataset "PubMedFull" ::setting "skewed_quantification_neighbor"}
   {:data.quantification_test_skew_tuplet_size 300}
   {::dataset "PubMedFull" ::setting "skewed_quantification_ppr"}
   {:data.quantification_test_skew_tuplet_size 300}
   {::dataset "ogbn-arxiv" ::setting "skewed_quantification_zipf"}
   {:data.quantification_test_skew_tuplet_size 1000}
   {::dataset "ogbn-arxiv" ::setting "skewed_quantification_neighbor"}
   {:data.quantification_test_skew_tuplet_size 1000}
   {::dataset "ogbn-arxiv" ::setting "skewed_quantification_ppr"}
   {:data.quantification_test_skew_tuplet_size 1000}
   {::setting "classification" ::dataset "ogbn-arxiv"}
   {:run.no_quantification "True"}
   ;; {::model "cuq_gcn"}
   ;; {:run.num_inits 2}
   ;; {::model "cuq_gat"}
   ;; {:run.num_inits 2}
   ;; {::model "cuq_gat" ::dataset "ogbn-arxiv"}
   ;; {:run.num_inits 10}
   })

(def default-datasets ["CoraML"
                       "CiteSeerFull"
                       "AmazonPhotos"
                       "AmazonComputers"
                       "PubMedFull"
                       #_"ogbn-arxiv"])

(def default-models (concat [#_"memory_quant"
                             "mlp"
                             "appnp"
                             "gcn"
                             "gat"
                             #_"matern_ggp"
                             #_"gdk"
                             #_"gpn"
                             #_"gpn_rw"
                             #_"gpn_lop"
                             #_"cuq_appnp"
                             #_"cuq_gcn"
                             #_"cuq_gat"]
                            #_(->> (keys models)
                                   (filter ;#(re-matches #"qgnn_.*_(\d+)$" %)
                                    #(re-matches #"qgnn_.*$" %))
                                   (sort-by (fn [model]
                                              (let [v [(if-let [[_ num] (re-matches #".*_(\d+)$" model)]
                                                         (parse-long num)
                                                         1000)
                                                       (condp #(str/includes? %2 %1) model
                                                         "_nn_" 0
                                                         "_mean_" 1)
                                                       (condp #(str/includes? %2 %1) model
                                                         "appnp" 0
                                                         "gcn" 1
                                                         "gat" 2)
                                                       (condp #(str/includes? %2 %1) model
                                                         "random" 0
                                                         "zipf" 1)]]
                                                #_(println model v)
                                                v))))))
(def default-settings ["classification"
                       #_"ood_loc"
                       #_"ood_features_normal"
                       #_"ood_features_ber"
                       "skewed_quantification_neighbor"
                       "skewed_quantification_ppr"
                       "skewed_quantification_zipf"])

(comment
  default-models
  nil)

;; Result post-processing
;; Some results need some post-processing to fix inconsistencies in how some results were labeled.
(defn transform-matern-ggp-results
  [results]
  (update-vals results
               (fn [{:keys [ood_detection_epistemic_auroc]
                     :as r}]
                 (merge r
                        {:ood_detection_epistemic_auroc nil
                         :ood_detection_epistemic_entropy_auroc
                         ood_detection_epistemic_auroc}))))

;; Utils

(defn config-transform
  [config]
  (if (::tuplet-size-factor config)
    (let [factor (* (::num-classes config)
                    (::tuplet-size-factor config))]
      (assoc config
             :data.quantification_train_tuplet_size factor
             :data.quantification_train_tuplet_oversampling_factor 100
             :data.quantification_val_tuplet_size factor
             :data.quantification_val_tuplet_oversampling_factor 100
             :data.quantification_test_tuplet_size factor
             :data.quantification_test_tuplet_oversampling_factor 100))
    config))

(defn run-config!
  [config & {:as overrides}]
  (let [args (keep (fn [[k v]]
                     (when-not (namespace k)
                       (str (name k) "=" (if (sequential? v) (str/join "," v) v))))
                   overrides)]
    (log/info "Running with" config (str/join " " args))
    (try
      (apply shell "python3 train_and_eval.py" "--force"
             "with" config args)
      (catch ExceptionInfo _
        (log/error "Run failed:"
                   "python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client train_and_eval.py --force with"
                   config (str/join " " args))))))

(defn build-config-path
  [_ model setting]
  (let [model-name (::name model)
        setting-name (::name setting)
        dir (cond
              (str/includes? model-name "gpn") "gpn"
              (or (str/includes? model-name "qgnn")
                  (str/includes? model-name "quant"))
              "quantification"
              :else "reference")
        path (str "configs/" dir "/"
                  (when (not= dir "quantification") (str setting-name "_"))
                  model-name ".yaml")]
    (assert (fs/exists? path) (str "Config path does not exist: " path))
    path))

(defn build-config-cli-params
  [dataset-name model-name setting-name overrides]
  (assert (contains? datasets dataset-name) (str "Dataset not found: " dataset-name))
  (assert (contains? models model-name) (str "Model not found: " model-name))
  (assert (contains? settings setting-name) (str "Setting not found: " setting-name))
  (let [dataset (merge {::name dataset-name
                        :data.dataset dataset-name}
                       (datasets dataset-name))
        model (merge {::name model-name} (models model-name))
        setting (merge {::name setting-name} (settings setting-name))
        combination-overrides
        (apply merge
               (for [d [nil [::dataset dataset-name]]
                     m [nil [::model model-name]]
                     s [nil [::setting setting-name]]
                     :let [c (combination-overrides (into {} (remove nil?) [d m s]))]
                     :when c]
                 c))
        config-dict (merge dataset model setting combination-overrides overrides)
        config-dict (config-transform config-dict)]
    [(build-config-path dataset model setting)
     (dissoc config-dict ::name)]))

(defn stringify-combination
  [dataset-name model-name setting-name]
  (str setting-name "/" dataset-name "/" model-name))

(defn- valid-var?
  [var]
  (cond
    (vector? var) (every? valid-var? var)
    (number? var) (not (Double/isNaN var))
    :else false))

(defn recurse
  [f x]
  (if (vector? x)
    (mapv (partial recurse f) x)
    (f x)))

(defn add-ses-to-var-map
  [var-map]
  (let [n (::num-samples var-map)]
    (into var-map
          (comp (filter #(-> % first name (str/ends-with? "_var")))
                (filter #(-> % second valid-var?))
                (mapcat (fn [[k v]]
                          (let [k (name k)
                                k-base (subs k 0 (- (count k) 4))
                                k-se (keyword (str k-base "_se"))
                                k-sd (keyword (str k-base "_sd"))]
                            (cond-> []
                              (not (contains? var-map k-se))
                              (conj [k-se (recurse #(math/sqrt (/ % n)) v)])
                              (not (contains? var-map k-sd))
                              (conj [k-sd (recurse #(math/sqrt %) v)]))))))
          var-map)))

(defn update-cached-results
  [results overrides]
  (let [num-samples (* (:run.num_splits overrides) (:run.num_inits overrides))
        test-num-samples num-samples
        #_(if (= (:data.quantification_test_skew overrides) "True")
            (* num-samples
               (::num-classes overrides)
               (:data.quantification_test_skew_repeats overrides))
            num-samples)]
    (into {}
          (map (fn [[k v]]
                 [k (-> v
                        (assoc ::num-samples
                               (if (= k :test) test-num-samples num-samples))
                        (add-ses-to-var-map))]))
          results)))

(defn get-results
  [dataset-name model-name setting-name overrides
   & {:keys [only-cached no-cache delete]
      :or {only-cached false no-cache false delete false}}]
  (assert (not (and only-cached no-cache))
          "only-cached and no-cache cannot be enabled at the same time.")
  (assert (not (and only-cached delete))
          "only-cached and delete cannot be enabled at the same time.")
  (let [combination-id (stringify-combination dataset-name
                                              model-name
                                              setting-name)
        results-path (str "results/" combination-id ".json")
        [_ config :as params]
        (build-config-cli-params dataset-name
                                 model-name
                                 setting-name
                                 (assoc overrides :run.results_path results-path))]
    (when (::skip config)
      (throw (ex-info (str "Skipped " combination-id ".")
                      {::cause :skip
                       :dataset-name dataset-name
                       :model-name model-name
                       :setting-name setting-name
                       :overrides overrides})))
    (if (and (not (:run.retrain overrides))
             (not (:run.reeval overrides))
             (not (:run.partial_reeval overrides))
             (not no-cache)
             (not delete)
             (fs/exists? results-path))
      (log/debug "Loading" combination-id "from cache...")
      (if only-cached
        (throw (ex-info (str "No cached results for " combination-id)
                        {::cause :no-cache
                         :dataset-name dataset-name
                         :model-name model-name
                         :setting-name setting-name
                         :overrides overrides}))
        (do
          (log/info (if delete "Deleting" "Running")
                    combination-id "...")
          (fs/create-dirs (fs/parent results-path))
          (apply run-config! params))))
    (when-not delete
      (let [results (json/parse-stream (io/reader results-path) true)
            result-transform (::result-transform config)
            results (if result-transform (result-transform results) results)
            results (update-cached-results results config)
            results (update-vals results #(apply dissoc % (::ignored-metrics config)))]
        results))))

(defn try-get-results
  [& args]
  (try
    (apply get-results args)
    (catch clojure.lang.ExceptionInfo e
      (if (= (::cause (ex-data e)) :skip)
        (log/debug (ex-message e))
        (log/error (ex-message e)))
      nil)))

;; Run experiments

(defn expand-globs
  [patterns candidate-matches]
  (into [] (comp (mapcat (fn [pattern]
                           (if (glob/glob? pattern)
                             (glob/glob pattern candidate-matches)
                             [pattern])))
                 (distinct))
        patterns))

(defn run-combinations!
  [dataset-names model-names setting-names overrides & {:as opts}]
  (doseq [dataset-name dataset-names
          model-name model-names
          setting-name setting-names]
    (try-get-results dataset-name model-name setting-name overrides opts)))

(defn print-grid
  [dataset-names model-names setting-names overrides]
  (println "Datasets:")
  (doseq [dataset dataset-names]
    (println "-" dataset))
  (println "\nModels:")
  (doseq [model model-names]
    (println "-" model))
  (println "\nSettings:")
  (doseq [setting setting-names]
    (println "-" setting))
  (println "\nOverrides:")
  (doseq [[k v] overrides]
    (println "-" k "=" v)))

(defn parse-override
  [override]
  (let [[k v] (str/split override #"=" 2)
        k (str/trim k)
        v (str/trim v)]
    (assert k)
    (assert v)
    [(keyword k) v]))

(defn run-eval!
  [{:keys [dataset model setting override
           dry retrain reeval partial-reeval reeval-metric
           only-cached cache delete]
    :or {dataset default-datasets
         model default-models
         setting default-settings}}]
  (let [dataset (expand-globs dataset (sort (keys datasets)))
        model (expand-globs model (sort (keys models)))
        setting (expand-globs setting (sort (keys settings)))
        default-config (cond-> {}
                         retrain (assoc :run.retrain true)
                         reeval (assoc :run.reeval true)
                         (or partial-reeval (seq reeval-metric))
                         (assoc :run.partial_reeval true)
                         (seq reeval-metric) (assoc :run.reeval_metrics reeval-metric)
                         delete (assoc :run.delete_run true))
        override (into default-config (map parse-override) override)]
    (print-grid dataset model setting override)
    (when delete
      (print "\nAre you sure you want to delete all results listed above? (y/N) ")
      (flush)
      (let [input (read-line)]
        (if (str/starts-with? (str/lower-case input) "y")
          (do
            (log/info "Will start deleting all selected results in 3s...")
            (Thread/sleep 1000)
            (log/info "Will start deleting all selected results in 2s...")
            (Thread/sleep 1000)
            (log/info "Will start deleting all selected results in 1s...")
            (Thread/sleep 1000)
            (log/info "Deleting..."))
          (do
            (log/error "Aborted.")
            (System/exit 1)))))
    (when-not dry
      (log/info (str (if delete "Deleting" "Starting")
                     " experiments ("
                     "only-cached=" only-cached ", "
                     "cache=" cache
                     ")..."))
      (Thread/sleep 500)
      (run-combinations! dataset model setting override
                         :only-cached only-cached
                         :no-cache (not cache)
                         :delete delete))
    (log/info "Done.")))

;; Accuracy-rejection tables

(defn get-acc-rej-curve
  [dataset-name model-name confidence-type uncertainty-type]
  (let [key (str "accuracy_rejection_" confidence-type "_confidence_"
                 uncertainty-type)
        mean-kw (keyword key)
        var-kw (keyword (str key "_var"))
        se-kw (keyword (str key "_se"))
        results (try-get-results dataset-name model-name
                                 "classification" {}
                                 :only-cached true)
        results (:test results)
        mean (mean-kw results)
        var (var-kw results)
        se (se-kw results)]
    (when mean
      {:mean mean, :var var, :se se})))

(defn get-acc-rej-curve-with-fallback
  [dataset-name model-name types]
  (first (eduction (keep #(apply get-acc-rej-curve dataset-name model-name %))
                   types)))

(defn run-acc-rej-table-gen!
  [dataset type]
  (let [types (case type
                "sample"
                [["sample" "epistemic"]
                 ["sample" "aleatoric"]]
                "prediction"
                [["prediction" "total"]
                 ["prediction" "epistemic"]
                 ["prediction" "aleatoric"]]
                "sample_total"
                [["sample" "total"]]
                "sample_total_entropy"
                [["sample" "total_entropy"]]
                "sample_aleatoric"
                [["sample" "aleatoric"]]
                "sample_aleatoric_entropy"
                [["sample" "aleatoric_entropy"]]
                "sample_epistemic"
                [["sample" "epistemic"]]
                "sample_epistemic_entropy"
                [["sample" "epistemic_entropy"]]
                "sample_epistemic_entropy_diff"
                [["sample" "epistemic_entropy_diff"]]
                "prediction_aleatoric"
                [["prediction" "aleatoric"]]
                "prediction_epistemic"
                [["prediction" "epistemic"]])
        curves
        (into {}
              (comp (map #(do [% (get-acc-rej-curve-with-fallback dataset %
                                                                  types)]))
                    (filter (comp :mean second)))
              default-models)
        model-names (keys curves)
        N (-> curves first second :mean count)
        head (str/join "," (into ["p"]
                                 (comp
                                  (map #(-> % models (::colname %)))
                                  (mapcat #(do [(str % "Mean")
                                                (str % "Var")
                                                (str % "SE")])))
                                 model-names))
        body (for [i (range N)
                   :let [p (double (/ i (dec N)))]]
               (str/join ","
                         (into [p]
                               (mapcat (fn [model]
                                         (let [{:keys [mean var se]}
                                               (curves model)]
                                           [(get mean i)
                                            (get var i 0)
                                            (get se i 0)]))
                                       model-names))))
        csv (str/join "\n" (cons head body))]
    (fs/create-dirs "tables")
    (spit (str "tables/acc_rej_" type "_" (-> dataset datasets (::colname dataset)) ".csv") csv)))

(defn run-acc-rej-tables-gen!
  [& _]
  (log/info "Generating accuracy-rejection tables...")
  (doseq [dataset default-datasets
          type ["sample" "prediction"
                "sample_total" "sample_total_entropy"
                "sample_aleatoric" "sample_aleatoric_entropy"
                "sample_epistemic" "sample_epistemic_entropy"
                "sample_epistemic_entropy_diff"
                "prediction_aleatoric" "prediction_epistemic"]]
    (run-acc-rej-table-gen! dataset type))
  (log/info "Done."))

;; ID-OOD table

(defn compute-certainty-change
  ([results uncertainty-type]
   (compute-certainty-change results uncertainty-type false))
  ([results uncertainty-type total-norm]
   (let [id-certainty (keyword (str "id_avg_sample_confidence_" uncertainty-type))
         ood-certainty (keyword (str "ood_avg_sample_confidence_" uncertainty-type))

         id-certainty (id-certainty results)
         ood-certainty (ood-certainty results)
         norm (if total-norm
                (:id_avg_sample_confidence_total_entropy results)
                id-certainty)]
     #_(println uncertainty-type ood-certainty id-certainty norm)
     (when (and ood-certainty norm)
       (if (not= (math/signum ood-certainty) (math/signum norm))
         (do (log/warn "Sign mismatch" {:uncertainty-type uncertainty-type
                                        :ood-certainty ood-certainty
                                        :norm norm
                                        :total-norm total-norm})
             nil)
         (dec (/ (- ood-certainty) (Math/abs norm))))))))

(defn round
  ([n places & {:keys [factor decimals sign leading-zero]
                :or {factor 1
                     decimals nil
                     sign false
                     leading-zero true}}]
   (when n
     (let [s (if (and (not leading-zero) (< (abs n) 1)) 1 0)
           decimals (or decimals (+ places s -1))
           scale (Math/pow 10 decimals)
           num (/ (Math/round (* n scale factor)) scale)
           res (format (str "%." decimals "f") num)
           l (+ places s 1)
           l (if (and (not sign) (pos? num)) l (inc l))
           l (min (count res) l)
           res (subs res s l)]
       (if (and sign (pos? num))
         (str "+" res)
         res)))))

(defn get-se
  [results metric]
  (let [se (keyword (str metric "_se"))]
    (se results 0)))

(defn get-metric
  [results best-results metric]
  (let [result (metric results)
        best-result (best-results metric)]
    [(round result 4 :decimals 2 :factor 100)
     (round (get-se results metric) 4 :decimals 2 :factor 100)
     (if (and result best-result (>= result best-result)) "1" "0")]))

(defn run-id-ood-table-gen!
  [& _]
  (log/info "Generating ID-OOD table...")
  (let [dataset-names default-datasets
        model-names default-models
        setting-names (rest default-settings)
        class-metrics [:accuracy]
        settings-metrics [:id_accuracy
                          :ood_accuracy
                          :ood_detection_total_auroc
                          :ood_detection_total_entropy_auroc
                          :ood_detection_aleatoric_auroc
                          :ood_detection_aleatoric_entropy_auroc
                          :ood_detection_epistemic_auroc
                          :ood_detection_epistemic_entropy_auroc
                          :ood_detection_epistemic_entropy_diff_auroc]
        cols (into ["id" "dataset" "model" "acc" "accSE" "accBest"]
                   (mapcat (fn [setting]
                             (let [setting (-> setting settings ::colname)]
                               (concat
                                (mapcat #(do [% (str % "SE") (str % "Best")])
                                        [(str setting "IdAcc")
                                         (str setting "OodAcc")
                                         (str setting "OodTotal")
                                         (str setting "OodTotalEntropy")
                                         (str setting "OodAleatoric")
                                         (str setting "OodAleatoricEntropy")
                                         (str setting "OodEpistemic")
                                         (str setting "OodEpistemicEntropy")
                                         (str setting "OodEpistemicEntropyDiff")])
                                [(str setting "TotalEntropyChange")
                                 (str setting "AleatoricEntropyChange")
                                 (str setting "EpistemicEntropyChange")
                                 (str setting "EpistemicEntropyDiffChange")]))))
                   setting-names)
        head (str/join "," cols)
        rows (for [dataset dataset-names
                   model model-names
                   :let [class-results
                         (-> (try-get-results dataset model "classification" {}
                                              :only-cached true)
                             :test)
                         setting-results (zipmap setting-names
                                                 (map (fn [setting]
                                                        (-> (try-get-results dataset model setting {}
                                                                             :only-cached true)
                                                            :test))
                                                      setting-names))]]
               {:dataset dataset
                :model model
                :class-results class-results
                :setting-results setting-results})
        metrics (concat (map #(do [:class-results %]) class-metrics)
                        (for [setting setting-names
                              metric settings-metrics]
                          [:setting-results setting metric]))
        row-groups (group-by :dataset rows)
        best-metric-groups
        (into {}
              (mapcat (fn [[group rows]]
                        (map (fn [metric]
                               (let [vals (keep #(get-in % metric) rows)
                                     best-val (apply max ##-Inf vals)]
                                 [(conj metric group) best-val]))
                             metrics)))
              row-groups)
        body (map-indexed
              (fn [i {:keys [dataset model class-results setting-results]}]
                (let [row [i
                           (-> dataset datasets (::inline-name dataset))
                           (-> model models (::inline-name model))]
                      class-best-results (fn [metric] (best-metric-groups [:class-results metric dataset]))
                      row (into row
                                (mapcat #(get-metric class-results class-best-results %))
                                class-metrics)]
                  (str/join ","
                            (into row
                                  (mapcat (fn [[setting results]]
                                            (let [best-results (fn [metric]
                                                                 (best-metric-groups [:setting-results setting
                                                                                      metric dataset]))
                                                  total-entropy-change (compute-certainty-change results "total_entropy")
                                                  aleatoric-entropy-change (compute-certainty-change results "aleatoric_entropy")
                                                  epistemic-entropy-change (compute-certainty-change results "epistemic_entropy")
                                                  epistemic-entropy-diff-change (compute-certainty-change results "epistemic_entropy_diff")]
                                              (concat (mapcat #(get-metric results best-results %)
                                                              settings-metrics)
                                                      (map #(round % 4 :factor 100 :sign true)
                                                           [total-entropy-change
                                                            aleatoric-entropy-change
                                                            epistemic-entropy-change
                                                            epistemic-entropy-diff-change])))))
                                  setting-results))))
              rows)
        csv (str/join "\n" (cons head body))]
    (log/info (str "Creating table with " (count cols) " columns..."))
    (fs/create-dirs "tables")
    (spit "tables/id_ood.csv" csv))
  (log/info "Done."))

;; Qunatification Tables

(defn run-ac-quantification-table-gen!
  [& _]
  (log/info "Generating AC quantification tables...")
  (doseq [:let [approach-dict {"cc"
                               ["cc"
                                "acc"
                                "maj_neighbor_pred_acc"
                                #_"weighted_dist_exp_acc"
                                #_"weighted_dist_avg_exp_10_pacc"
                                #_"ppr_group_acc"
                                "ppr_group_int_9_acc"
                                #_"ppr_pairs_acc"
                                #_"weighted_dist_neighbor_exp_acc"
                                #_"weighted_dist_neighbor_avg_exp_10_acc"
                                #_"ppr_neighbor_group_acc"
                                "ppr_neighbor_group_int_9_acc"]
                               "pcc"
                               ["pcc"
                                "pacc"
                                "maj_neighbor_pred_pacc"
                                #_"weighted_dist_exp_pacc"
                                #_"ppr_group_int_pacc"
                                "ppr_group_int_9_pacc"
                                #_"ppr_pairs_pacc"
                                #_"weighted_dist_neighbor_exp_pacc"
                                #_"ppr_neighbor_group_pacc"
                                #_"ppr_neighbor_group_int_pacc"
                                "ppr_neighbor_group_int_9_pacc"]
                               "pcc_sp"
                               ["pcc"
                                "pacc"
                                "maj_neighbor_pred_pacc"
                                "weighted_dist_exp_30_pacc"
                                #_"ppr_group_int_pacc"
                                #_"ppr_group_int_9_pacc"
                                #_"ppr_pairs_pacc"
                                "weighted_dist_neighbor_exp_30_pacc"
                                #_"ppr_neighbor_group_pacc"
                                #_"ppr_neighbor_group_int_pacc"
                                "ppr_neighbor_group_int_9_pacc"]}]
          approach-key ["cc" "pcc" "pcc_sp"]]
    (let [dataset-names default-datasets
          model-names (if (= approach-key "pcc_sp")
                        ["memory_quant"
                         "mlp"
                         "gat"
                         "gcn"
                         "appnp"]
                        ["memory_quant"
                         "ego_net"
                         "mlp"
                         "gat"
                         "gcn"
                         "appnp"])
          settings [#_"classification"
                    "skewed_quantification_zipf"
                    "skewed_quantification_neighbor"
                    "skewed_quantification_ppr"]
          aggregation-approaches (approach-dict approach-key)
          quantification-metrics ["ae" "rae" #_"kld"]
          results (into {}
                        (for [ds dataset-names
                              model model-names
                              setting settings]
                          [[ds model setting]
                           (try-get-results ds model setting {}
                                            :only-cached true)]))
          rows
          (for [setting settings
                model model-names
                :let [aggregation-approaches
                      (if (or (str/includes? model "quant") (str/includes? model "qgnn"))
                        ["tuplets"]
                        aggregation-approaches)
                      aggregation-approaches
                      (if (= model "ego_net")
                        (->> aggregation-approaches
                             (remove #(str/includes? % "weighted_dist"))
                             (map #(str/replace % #"group_int_9" "group_int")))
                        aggregation-approaches)]
                approach aggregation-approaches
                :let [model-name (-> model models ::inline-name)
                      model-suffix
                      ({"skewed_quantification_zipf" "PPS"
                        "skewed_quantification_neighbor" "BFS"
                        "skewed_quantification_ppr" "RW"}
                       setting)
                      rename (fn [s]
                               ({"CC" "CC"
                                 "PCC" "PCC"
                                 "ACC" "ACC"
                                 "PACC" "PACC"} s s))
                      approach-name
                      (if (not= approach "tuplets")
                        (-> approach
                            (str/replace #"(maj_)?neighbor" "neigh")
                            (str/replace #"weighted_dist" "sp")
                            (str/replace #"ppr" (if (= approach-key "pcc_sp") "PPR" "SIS"))
                            (str/replace #"group|exp|pred|int|avg|\d+" "")
                            str/upper-case
                            (str/replace "_" " ")
                            (str/trim)
                            #_(rename)
                            #_(str/replace #"\s+(P)?ACC" ""))
                        (if (= model "memory_quant")
                          "MLPE"
                          (str (if (str/includes? model "mean")
                                 "M" "NN")
                               "+"
                               (if (str/includes? model "random")
                                 "R" "Z"))))
                      row (ordered-map "model" model-name
                                       "model_suffix" model-suffix
                                       "approach" approach-name)
                      row (into row
                                (mapcat (fn [{:keys [dataset metric]}]
                                          (let [colname (str (-> dataset datasets ::colname) "_" metric)
                                                results (:test (results [dataset model setting]))
                                                num-samples (::num-samples results)
                                                kw-base (str "quantification_"
                                                             approach "_" metric)
                                                kw-base (if (and (str/includes? approach "int_9")
                                                                 (nil? (get results (keyword kw-base))))
                                                          (str "quantification_"
                                                               (str/replace approach #"int_9" "int")
                                                               "_" metric)
                                                          kw-base)
                                                mean (get results
                                                          (keyword kw-base))
                                                se (get results
                                                        (keyword (str kw-base "_se")))
                                                sd (get results
                                                        (keyword (str kw-base "_sd")))]
                                            [[colname mean]
                                             [(str colname "_se") se]
                                             [(str colname "_best") {:mean mean :sd sd :n num-samples}]])))
                                (for [ds dataset-names
                                      metric quantification-metrics]
                                  {:dataset ds
                                   :metric metric}))]]
            row)
          group-fn (juxt #(get % "model") #(get % "model_suffix"))
          row-groups (group-by group-fn rows)
          metrics (for [ds dataset-names
                        metric quantification-metrics]
                    (str (-> ds datasets ::colname) "_" metric))
          best-metric-groups
          (into {}
                (mapcat (fn [[group rows]]
                          (map (fn [metric]
                                 (let [vals (sequence (comp (map #(get % (str metric "_best")))
                                                            (filter :mean))
                                                      rows)
                                       best-val (apply min-key :mean vals)
                                       row->rank (into {}
                                                       (map-indexed (fn [r row] [row (inc r)]))
                                                       (sort-by #(get % metric) rows))]
                                   [[metric group] {:best best-val :ranks row->rank}]))
                               metrics)))
                row-groups)
          rows (map (fn [row]
                      (into row
                            (mapcat (fn [metric]
                                      (let [{{best-mean :mean, best-sd :sd, best-n :n} :best
                                             row->rank :ranks}
                                            (best-metric-groups [metric (group-fn row)])
                                            {:keys [mean sd n]} (get row (str metric "_best"))]
                                        [[(str metric "_best")
                                          (cond
                                            (or (not (and mean best-mean))
                                                (= (row "approach") "MLPE"))
                                            false
                                            (and (zero? sd) (zero? best-sd))
                                            (<= mean best-mean)
                                            :else
                                            (not (t/significant? (t/t-test {:mean best-mean :sd best-sd :n best-n}
                                                                           {:mean mean :sd sd :n n})
                                                                 0.05 :<)))]
                                         [(str metric "_rank") (get row->rank row)]])))
                            metrics))
                    rows)
          rows (map (fn [row]
                      (into row
                            (map (fn [q-metric]
                                   (let [ranks (map #(get row (str (-> % datasets ::colname) "_" q-metric "_rank"))
                                                    dataset-names)
                                         avg-rank (/ (apply + ranks) (count ranks))]
                                     [(str q-metric "_rank") avg-rank])))
                            quantification-metrics))
                    rows)
          best-rank-groups
          (into {}
                (mapcat (fn [[group rows]]
                          (map (fn [q-metric]
                                 [[q-metric group]
                                  (transduce (map #(get % (str q-metric "_rank"))) min ##Inf rows)])
                               quantification-metrics)))
                (group-by group-fn rows))
          rows (map (fn [row]
                      (let [group (group-fn row)
                            group-size (count (get row-groups group))]
                        (into (assoc row "group_size" group-size)
                              (mapcat (fn [q-metric]
                                        (let [best-rank (best-rank-groups [q-metric (group-fn row)])
                                              rank (get row (str q-metric "_rank"))]
                                          (if (= (row "approach") "MLPE")
                                            [[(str q-metric "_rank") nil]
                                             [(str q-metric "_rank_best") false]]
                                            [[(str q-metric "_rank") (round rank 3 :decimals 1)]
                                             [(str q-metric "_rank_best") (<= rank best-rank)]]))))
                              quantification-metrics)))
                    rows)
          cols (map normalize-colname (keys (first rows)))
          head (str/join "," cols)
          rows (map (fn [row] (str/join "," (map #(cond
                                                    (boolean? %) (if % "1" "0")
                                                    (integer? %) (str %)
                                                    (number? %) (round % 4 :leading-zero false)
                                                    :else %)
                                                 (vals row)))) rows)
          csv (str/join "\n" (cons head rows))]
      (log/info "Creating table for approach" approach-key "with" (count cols) "columns...")
      (fs/create-dirs "tables")
      (spit (str "tables/" approach-key ".csv") csv)
      (log/info "Done."))))

(defn run-dm-quantification-table-gen!
  [& _]
  (log/info "Generating DM quantification tables...")
  (let [dataset-names default-datasets
        model-names ["memory_quant"
                     "ego_net"
                     "mlp"
                     "gat"
                     "gcn"
                     "appnp"]
        settings [#_'"classification"
                  "skewed_quantification_zipf"
                  "skewed_quantification_neighbor"
                  "skewed_quantification_ppr"]
        aggregation-approaches [#_"cc"
                                #_"acc"
                                "pcc"
                                "pacc"
                                #_"weighted_dist_exp_acc"
                                "ppr_group_int_9_pacc"
                                #_"ppr_pairs_acc"
                                "kdey"
                                #_"kdey_ppr_int_1_10_10"
                                #_"kdey_ppr_int_2_10_10"
                                #_"kdey_ppr_int_5_10_10"
                                #_"kdey_ppr_int_8_10_10"
                                "kdey_ppr_int_9_10_10"
                                #_"kdey_ppr_10_10"
                                #_"kdey_dist_avg_inv_dist_part"
                                #_"kdey_dist_avg_inv_avg_deg"
                                #_"kdey_dist_avg_inv_1"
                                #_"kdey_dist_avg_exp_2"
                                #_"kdey_dist_avg_exp_5"]
        quantification-metrics ["ae" "rae" "kld"]
        results (into {}
                      (for [ds dataset-names
                            model model-names
                            setting settings]
                        [[ds model setting]
                         (try-get-results ds model setting {}
                                          :only-cached true)]))
        rows
        (for [setting settings
              model model-names
              :let [aggregation-approaches
                    (if (or (str/includes? model "quant") (str/includes? model "qgnn"))
                      ["tuplets"]
                      (if (= setting "skewed_quantification_neighbor")
                        #_(concat aggregation-approaches ["kdey_dist_avg_exp_5"])
                        aggregation-approaches
                        aggregation-approaches))]
              approach aggregation-approaches
              :let [model-name (-> model models ::inline-name)
                    model-suffix
                    ({"skewed_quantification_zipf" "PPS"
                      "skewed_quantification_neighbor" "BFS"
                      "skewed_quantification_ppr" "RW"}
                     setting)
                    approach-name
                    (if (not= approach "tuplets")
                      (-> approach
                          (str/replace "ppr_group_int_pacc" "pacc_ppr_int_5")
                          (str/replace #"group|pred|inv|(?<!int_|exp_)\d+|avg_deg|dist_part" "")
                          (str/replace #"(weighted_)?dist(_avg)?" "SP")
                          (str/replace #"ppr(?!_int)" "PPR")
                          (str/replace #"ppr" "PPR")
                          #_(str/replace #"(int_|exp_)(\d+)" "0.$2")
                          (str/replace #"(int_|exp_)(\d+)" "")
                          str/upper-case
                          (str/replace "_" " ")
                          (str/trim)
                          (str/replace #"KDEY" "KDEy"))
                      (if (= model "memory_quant")
                        "MLPE"
                        (str (if (str/includes? model "mean")
                               "M" "NN")
                             "+"
                             (if (str/includes? model "random")
                               "R" "Z"))))
                    row (ordered-map "model" model-name
                                     "model_suffix" model-suffix
                                     "approach" approach-name)
                    row (into row
                              (mapcat (fn [{:keys [dataset metric]}]
                                        (let [colname (str (-> dataset datasets ::colname) "_" metric)
                                              results (:test (results [dataset model setting]))
                                              num-samples (::num-samples results)
                                              kw-base (str "quantification_"
                                                           approach "_" metric)
                                              kw-base (if (and (str/includes? approach "int_9")
                                                               (nil? (get results (keyword kw-base))))
                                                        (str "quantification_"
                                                             (str/replace approach #"int_9"
                                                                          (if (str/includes? approach "kdey")
                                                                            "int_5" "int"))
                                                             "_" metric)
                                                        kw-base)
                                              mean (get results
                                                        (keyword kw-base))
                                              se (get results
                                                      (keyword (str kw-base "_se")))
                                              sd (get results
                                                      (keyword (str kw-base "_sd")))]
                                          [[colname mean]
                                           [(str colname "_se") se]
                                           [(str colname "_best") {:mean mean :sd sd :n num-samples}]])))
                              (for [ds dataset-names
                                    metric quantification-metrics]
                                {:dataset ds
                                 :metric metric}))]]
          row)
        group-fn (juxt #(get % "model") #(get % "model_suffix"))
        row-groups (group-by group-fn rows)
        metrics (for [ds dataset-names
                      metric quantification-metrics]
                  (str (-> ds datasets ::colname) "_" metric))
        best-metric-groups
        (into {}
              (mapcat (fn [[group rows]]
                        (map (fn [metric]
                               (let [vals (sequence (comp (map #(get % (str metric "_best")))
                                                          (filter :mean))
                                                    rows)
                                     best-val (if (seq vals)
                                                (apply min-key :mean vals)
                                                ##-Inf)
                                     row->rank (into {}
                                                     (map-indexed (fn [r row] [row (inc r)]))
                                                     (sort-by #(get % metric) rows))]
                                 [[metric group] {:best best-val :ranks row->rank}]))
                             metrics)))
              row-groups)
        rows (map (fn [row]
                    (let [group (group-fn row)
                          group-size (count (get row-groups group))]
                      (into (assoc row "group_size" group-size)
                            (mapcat (fn [metric]
                                      (let [{{best-mean :mean, best-sd :sd, best-n :n} :best
                                             row->rank :ranks}
                                            (best-metric-groups [metric group])
                                            {:keys [mean sd n]} (get row (str metric "_best"))]
                                        [[(str metric "_best")
                                          (cond
                                            (or (not (and mean best-mean))
                                                (= (row "approach") "MLPE"))
                                            false
                                            (and (zero? sd) (zero? best-sd))
                                            (<= mean best-mean)
                                            :else
                                            (not (t/significant? (t/t-test {:mean best-mean :sd best-sd :n best-n}
                                                                           {:mean mean :sd sd :n n})
                                                                 0.05 :<)))]
                                         [(str metric "_rank") (get row->rank row)]])))
                            metrics)))
                  rows)
        rows (map (fn [row]
                    (into row
                          (map (fn [q-metric]
                                 (let [ranks (map #(get row (str (-> % datasets ::colname) "_" q-metric "_rank"))
                                                  dataset-names)
                                       avg-rank (/ (apply + ranks) (count ranks))]
                                   [(str q-metric "_rank") avg-rank])))
                          quantification-metrics))
                  rows)
        best-rank-groups
        (into {}
              (mapcat (fn [[group rows]]
                        (map (fn [q-metric]
                               [[q-metric group]
                                (transduce (map #(get % (str q-metric "_rank"))) min ##Inf rows)])
                             quantification-metrics)))
              (group-by group-fn rows))
        rows (map (fn [row]
                    (into row
                          (mapcat (fn [q-metric]
                                    (let [best-rank (best-rank-groups [q-metric (group-fn row)])
                                          rank (get row (str q-metric "_rank"))]
                                      (if (= (row "approach") "MLPE")
                                        [[(str q-metric "_rank") nil]
                                         [(str q-metric "_rank_best") false]]
                                        [[(str q-metric "_rank") (round rank 3 :decimals 1)]
                                         [(str q-metric "_rank_best") (<= rank best-rank)]]))))
                          quantification-metrics))
                  rows)
        cols (map normalize-colname (keys (first rows)))
        head (str/join "," cols)
        rows (map (fn [row] (str/join "," (map #(cond
                                                  (boolean? %) (if % "1" "0")
                                                  (integer? %) (str %)
                                                  (number? %) (round % 4 :leading-zero false)
                                                  :else %)
                                               (vals row)))) rows)
        csv (str/join "\n" (cons head rows))]
    (log/info "Creating table with" (count cols) "columns...")
    (fs/create-dirs "tables")
    (spit "tables/results.csv" csv)
    (log/info "Done.")))

(defn generate-quantification-tuplet-size-comp-table-gen!
  [dataset]
  (let [model-names ["qgnn_appnp" "qgnn_gcn" "qgnn_gat"]
        tuplet-aggregations ["nn" "mean"]
        sampling-strategies ["random" "zipf"]
        quantification-metrics ["ae" "rae" "kld" "jsd"]
        setting "skewed_quantification_zipf"
        row-dicts
        (for [tuplet-size [1 2 4 8 16 32]
              :let [row-dict
                    (into (ordered-map "tuplet-size" tuplet-size)
                          (for [model model-names
                                agg tuplet-aggregations
                                sampling sampling-strategies
                                metric quantification-metrics
                                :let [full-model (str model "_" agg "_" sampling "_" tuplet-size)
                                      colname (str model "_" agg "_" sampling "_" metric)
                                      results (:test (try-get-results dataset full-model setting {}
                                                                      :only-cached true))]]
                            [colname (get results (keyword (str "quantification_tuplets_" metric)))]))]]
          row-dict)
        cols (map normalize-colname (keys (first row-dicts)))
        head (str/join "," cols)
        rows (map (fn [row] (str/join "," (map #(if (and (number? %) (not (integer? %)))
                                                  (round % 4 :leading-zero false) %)
                                               (vals row))))
                  row-dicts)
        csv (str/join "\n" (cons head rows))]
    (log/info "Creating table with" (count cols) "columns...")
    (fs/create-dirs "tables/quantification_tuplet_size_comp")
    (spit (str "tables/quantification_tuplet_size_comp/" dataset ".csv") csv)))

(defn run-quantification-tuplet-size-comp-table-gen!
  [& _]
  (log/info "Generating quantification tuplet size comparison table for dataset...")
  (let [dataset-names default-datasets]
    (doseq [dataset dataset-names]
      (generate-quantification-tuplet-size-comp-table-gen! dataset)))
  (log/info "Done."))

(defn run-quantification-acc-comp-table-gen!
  [& _]
  (log/info "Generating quantification accuracy comparison table...")
  (fs/create-dirs "tables/quantification_acc_comp")
  (doseq [ds default-datasets
          setting ["classification"
                   "skewed_quantification_zipf"
                   "skewed_quantification_neighbor"
                   "skewed_quantification_ppr"]]
    (let [setting-last (last (str/split setting #"_"))
          model-names [#_"ego_net"
                       "mlp"
                       "gat"
                       "gcn"
                       "appnp"]
          q-metrics ["ae" "rae" "kld"]
          adjusted_methods
          [; Adjusted maj neighbor
           "maj_neighbor_pred"
           "maj_neighbor_true"
           "maj_neighbor_full"
                                         ; Adjusted SIS
           "weighted_dist_exp_1"
           "weighted_dist_exp_2"
           "weighted_dist_exp_3"
           "weighted_dist_exp_4"
           "weighted_dist_exp" ; 0.5 is the default
           "weighted_dist_exp_6"
           "weighted_dist_exp_7"
           "weighted_dist_exp_8"
           "weighted_dist_exp_9"
           "weighted_dist_exp_10"
           "weighted_dist_exp_15"
           "weighted_dist_exp_20"
           "weighted_dist_exp_25"
           "weighted_dist_exp_30"
           "weighted_dist_inv_dist_part"
           "weighted_dist_inv_avg_deg"
           "weighted_dist_inv_5"
           "weighted_dist_inv_10"
           "weighted_dist_inv_20"
           "weighted_dist_avg_exp_1"
           "weighted_dist_avg_exp_2"
           "weighted_dist_avg_exp_3"
           "weighted_dist_avg_exp_4"
           "weighted_dist_avg_exp" ; 0.5 is the default
           "weighted_dist_avg_exp_6"
           "weighted_dist_avg_exp_7"
           "weighted_dist_avg_exp_8"
           "weighted_dist_avg_exp_9"
           "weighted_dist_avg_exp_10"
           "weighted_dist_avg_exp_15"
           "weighted_dist_avg_exp_20"
           "weighted_dist_avg_exp_25"
           "weighted_dist_avg_exp_30"
           "weighted_dist_avg_inv_dist_part"
           "weighted_dist_avg_inv_avg_deg"
           "weighted_dist_avg_inv_5"
           "weighted_dist_avg_inv_10"
           "weighted_dist_avg_inv_20"

           "ppr_group"
           "ppr_group_exp"
           "ppr_group_int_1"
           "ppr_group_int_2"
           "ppr_group_int_3"
           "ppr_group_int_4"
           "ppr_group_int"
           "ppr_group_int_6"
           "ppr_group_int_7"
           "ppr_group_int_8"
           "ppr_group_int_9"
           "ppr_pairs"
           "ppr_pairs_exp"
           "ppr_pairs_int"
                                         ; Adjusted maj neighbor + SIS
           "weighted_dist_neighbor_exp_1"
           "weighted_dist_neighbor_exp_2"
           "weighted_dist_neighbor_exp_3"
           "weighted_dist_neighbor_exp_4"
           "weighted_dist_neighbor_exp" ; 0.5 is the default
           "weighted_dist_neighbor_exp_6"
           "weighted_dist_neighbor_exp_7"
           "weighted_dist_neighbor_exp_8"
           "weighted_dist_neighbor_exp_9"
           "weighted_dist_neighbor_exp_10"
           "weighted_dist_neighbor_exp_15"
           "weighted_dist_neighbor_exp_20"
           "weighted_dist_neighbor_exp_25"
           "weighted_dist_neighbor_exp_30"
           "weighted_dist_neighbor_inv_dist_part"
           "weighted_dist_neighbor_inv_avg_deg"
           "weighted_dist_neighbor_inv_5"
           "weighted_dist_neighbor_inv_10"
           "weighted_dist_neighbor_inv_20"
           "weighted_dist_neighbor_avg_exp_1"
           "weighted_dist_neighbor_avg_exp_2"
           "weighted_dist_neighbor_avg_exp_3"
           "weighted_dist_neighbor_avg_exp_4"
           "weighted_dist_neighbor_avg_exp" ; 0.5 is the default
           "weighted_dist_neighbor_avg_exp_6"
           "weighted_dist_neighbor_avg_exp_7"
           "weighted_dist_neighbor_avg_exp_8"
           "weighted_dist_neighbor_avg_exp_9"
           "weighted_dist_neighbor_avg_exp_10"
           "weighted_dist_neighbor_avg_exp_15"
           "weighted_dist_neighbor_avg_exp_20"
           "weighted_dist_neighbor_avg_exp_25"
           "weighted_dist_neighbor_avg_exp_30"
           "weighted_dist_neighbor_avg_inv_dist_part"
           "weighted_dist_neighbor_avg_inv_avg_deg"
           "weighted_dist_neighbor_avg_inv_5"
           "weighted_dist_neighbor_avg_inv_10"
           "weighted_dist_neighbor_avg_inv_20"

           "ppr_neighbor_group"
           "ppr_neighbor_group_exp"
           "ppr_neighbor_group_int_1"
           "ppr_neighbor_group_int_2"
           "ppr_neighbor_group_int_3"
           "ppr_neighbor_group_int_4"
           "ppr_neighbor_group_int"
           "ppr_neighbor_group_int_6"
           "ppr_neighbor_group_int_7"
           "ppr_neighbor_group_int_8"
           "ppr_neighbor_group_int_9"]
          adjusted_methods (mapcat #(do [(str % "_acc")
                                         (str % "_pacc")])
                                   adjusted_methods)
          methods (concat ["cc" "pcc" "acc" "pacc"]
                          adjusted_methods
                          [; DM
                           "dmy" "kdey"
                                                         ; DM SIS
                           "kdey_dist_avg_inv_1"
                           "kdey_dist_avg_inv_avg_deg"
                           "kdey_dist_avg_inv_dist_part"
                           "kdey_dist_avg_exp_1"
                           "kdey_dist_avg_exp_2"
                           "kdey_dist_avg_exp_5"
                           "kdey_ppr_10_10"
                           "kdey_ppr_int_1_10_10"
                           "kdey_ppr_int_2_10_10"
                           "kdey_ppr_int_3_10_10"
                           "kdey_ppr_int_4_10_10"
                           "kdey_ppr_int_5_10_10"
                           "kdey_ppr_int_6_10_10"
                           "kdey_ppr_int_7_10_10"
                           "kdey_ppr_int_8_10_10"
                           "kdey_ppr_int_9_10_10"])
          metric-names
          (concat ["accuracy"]
                  (for [m methods
                        qm q-metrics]
                    (str "quantification_" m "_" qm)))
          results (into {}
                        (for [model model-names]
                          [model
                           (try-get-results ds model setting {}
                                            :only-cached true)]))
          mlpe-results (:test (try-get-results ds "memory_quant" setting {}
                                               :only-cached true))
          baseline (into {}
                         (map (fn [m]
                                [(str "quantification_mlpe_" m)
                                 (get mlpe-results (keyword (str "quantification_tuplets_" m)))]))
                         q-metrics)
          rows (for [[model i] (map vector model-names (range))
                     :let [row (ordered-map
                                "i" i
                                "model" (-> model models ::inline-name))
                           row (merge row baseline)]]
                 (into row
                       cat
                       (for [metric metric-names
                             :let [results (:test (results model))
                                   mean (get results (keyword metric))
                                   se (get results (keyword (str metric "_se")))]]
                         [[metric mean]
                          [(str metric "_se") se]])))
          cols (map (comp csk/->camelCase name) (keys (first rows)))
          head (str/join "," cols)
          rows (map (fn [row] (str/join "," (vals row))) rows)
          csv (str/join "\n" (cons head rows))]
      (log/info "Creating table with" (count cols) "columns...")
      (spit (str "tables/quantification_acc_comp/" setting-last "_" ds ".csv") csv))))

;; CLI

(def CLI-CONFIGURATION
  {:command "gpn-extensions"
   :description "An evaluation script."
   :version "0.1.0"
   :subcommands [{:command "eval"
                  :description "Run experiments."
                  :opts [{:as "Datasets"
                          :option "dataset"
                          :short "d"
                          :type :string
                          :multiple true}
                         {:as "Models"
                          :option "model"
                          :short "m"
                          :type :string
                          :multiple true}
                         {:as "Settings"
                          :option "setting"
                          :short "s"
                          :type :string
                          :multiple true}
                         {:as "Overrides"
                          :option "override"
                          :short "o"
                          :type :string
                          :multiple true}
                         {:as "Dry Run"
                          :option "dry"
                          :default false
                          :type :with-flag}
                         {:as "Retrain Models"
                          :option "retrain"
                          :default false
                          :type :with-flag}
                         {:as "Reevaluate Models"
                          :option "reeval"
                          :default false
                          :type :with-flag}
                         {:as "Partial Reevaluation"
                          :option "partial-reeval"
                          :default false
                          :type :with-flag}
                         {:as "Reevaluate Metric"
                          :option "reeval-metric"
                          :type :string
                          :multiple true}
                         {:as "Only Cached"
                          :option "only-cached"
                          :default false
                          :type :with-flag}
                         {:as "No Cache"
                          :option "cache"
                          :default true
                          :type :with-flag}
                         {:as "Delete existing models and results"
                          :option "delete"
                          :default false
                          :type :with-flag}]
                  :runs run-eval!}
                 {:command "acc-rej-tables"
                  :description "Generate acc-rej CSVs."
                  :runs run-acc-rej-tables-gen!}
                 {:command "id-ood-table"
                  :description "Generate ID-OOD CSV."
                  :runs run-id-ood-table-gen!}
                 {:command "ac-quant-table"
                  :description "Generate AC quantification CSV."
                  :runs run-ac-quantification-table-gen!}
                 {:command "dm-quant-table"
                  :description "Generate DM quantification CSV."
                  :runs run-dm-quantification-table-gen!}
                 {:command "quant-tuplet-size-table"
                  :description "Generate quantification tuplet size comparison CSV."
                  :runs run-quantification-tuplet-size-comp-table-gen!}
                 {:command "quant-acc-table"
                  :description "Generate quantification accuracy comparison CSV."
                  :runs run-quantification-acc-comp-table-gen!}]})

(defn -main
  [& args]
  (cli/run-cmd (rest args) CLI-CONFIGURATION))

(comment
  (run-config! "configs/gpn/classification_gpn_16.yaml"
               :data.dataset "CoraML")
  (cli/run-cmd* [] CLI-CONFIGURATION)

  (run-acc-rej-tables-gen!)

  (doseq [path (fs/glob "." "results/classification/*/qgnn_*.json")
          :let [new-path (fs/path (fs/parent path) (str/replace (fs/file-name path)
                                                                #"\.json"
                                                                "_nn_random.json"))]]
    (println new-path)
    (fs/move path new-path))
  nil)
