(ns ad.core
  (:require [clojure.algo.generic.functor :as f])
  (:require [clojure.algo.generic.arithmetic :as a])
  (:require [clojure.algo.generic.math-functions :as m]))

(defrecord D [result perturbations])

(defn lift [x perturbations]
  (->D x (f/fmap (constantly 0) perturbations)))

(defmethod a/+ [D D] [{x :result xs' :perturbations}
                      {y :result ys' :perturbations}]
  (->D (a/+ x y)
       (merge-with a/+ xs' ys')))

(defmethod a/+ [D java.lang.Number] [x y]
  (a/+ x (lift y (:perturbations x))))

(defmethod a/+ [java.lang.Number D] [x y]
  (a/+ (lift x (:perturbations y)) y))

(defmethod a/* [D D] [{x :result xs' :perturbations}
                      {y :result ys' :perturbations}]
  (->D (a/* x y)
       (merge-with (fn [x' y']
                     (a/+ (a/* x' y) (a/* x y')))
                   xs' ys')))

(defmethod a/* [D java.lang.Number] [x y]
  (a/* x (lift y (:perturbations x))))

(defmethod a/* [java.lang.Number D] [x y]
  (a/* (lift x (:perturbations y)) y))

(defmethod a/- D [{x :result xs' :perturbations}]
  (->D (a/- x)
       (f/fmap a/- xs')))

(defn reciprocal [x]
  ((a/qsym a /) x))

(def div (a/qsym a /))

(a/defmethod* a / D [{x :result xs' :perturbations}]
  (->D (reciprocal x)
       (f/fmap (fn [x']
                 (div (a/- x') x x))
               xs')))

(defmethod m/sin D [{x :result xs' :perturbations}]
  (->D (m/sin x)
       (f/fmap (fn [x']
                 (a/* x' (m/cos x)))
               xs')))

(defmethod m/cos D [{x :result xs' :perturbations}]
  (->D (m/cos x)
       (f/fmap (fn [x']
                 (a/* (a/- x') (m/sin x)))
               xs')))

(defmethod m/exp D [{x :result xs' :perturbations}]
  (->D (m/exp x)
       (f/fmap (fn [x']
                 (a/* x' (m/exp x)))
               xs')))

(defmethod m/log D [{x :result xs' :perturbations}]
  (->D (m/log x)
       (f/fmap (fn [x']
                 (div x' x))
               xs')))

(defmethod m/abs D [{x :result xs' :perturbations}]
  (->D (m/abs x)
       (f/fmap (fn [x']
                 (a/* x' (m/sgn x)))
               xs')))

(defmethod m/pow [D java.lang.Long] [{x :result xs' :perturbations} k]
  (->D (m/pow x k)
       (f/fmap (fn [x']
                 (a/* k (m/pow x (dec k)) x'))
               xs')))

(defn diff [f x]
  (let [e (new Object)
        {:keys [perturbations]} (f (->D x {e 1}))]
    (get perturbations e)))

(defn diff' [f x]
  [(f x) (diff f x)])

(defn diff-f [f x]
  (let [e (new Object)]
    (map #(get (:perturbations %) e) (f (->D x {e 1})))))

(defn diff-f' [f x]
  [(f x) (diff-f f x)])

(defn grad-seeds [n]
  (if (<= n 1)
    [[1]]
    (cons (cons 1 (repeat (dec n) 0))
          (map (partial cons 0) (grad-seeds (dec n))))))

(defn grad [f x & xs]
  (let [xs (cons x xs)]
    (map (fn [seeds]
           (let [e (new Object)
                 {:keys [perturbations]}
                 (apply f (map (fn [x seed]
                                 (->D x {e seed}))
                               xs seeds))]
             (get perturbations e)))
         (grad-seeds (count xs)))))

(defn grad' [f x & xs]
  [(apply f x xs) (apply grad f x xs)])

(defn gradient-descent [f x0 & xs0]
  (letfn [(step [[x fx gx step-size i]]
            (cond (zero? step-size) nil
                  (every? zero? gx) nil
                  :else
                  (let [x' (map (fn [xi gxi]
                                  (- xi (* step-size gxi)))
                                x gx)
                        [fx' gx'] (apply grad' f x')]
                    (cond (> fx' fx)
                          (recur [x fx gx (/ step-size 2) 0])
                          (= i 10)
                          [x' fx' gx' (* step-size 2) 0]
                          :else
                          [x' fx' gx' step-size (inc i)]))))]
    (let [[fx gx] (apply grad' f x0 xs0)]
      (map first
           (take-while (complement nil?)
                       (iterate step [(cons x0 xs0) fx gx 0.1 0]))))))

(defn rosenbrock [x y]
  (a/+ (m/pow (a/- 1 x) 2)
       (a/* 100 (m/pow (a/- y (m/pow x 2)) 2))))
