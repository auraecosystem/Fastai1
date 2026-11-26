"""Regression checks for the historical AI-model examples.

Optional ML frameworks are imported only when their corresponding demo is
called. Test collection never downloads models or initializes a deep-learning
runtime.
"""


def reactive_machine(input_value):
    """A deterministic rule-based example."""
    return "Positive" if input_value > 0 else "Negative"


def limited_memory_model():
    """Train the original sklearn demonstration on a generated dataset."""
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split

    features, labels = make_classification(n_samples=1000, n_features=20, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    model = LinearRegression()
    model.fit(x_train, y_train)
    predictions = (model.predict(x_test) > 0.5).astype(int)
    return accuracy_score(y_test, predictions)


def theory_of_mind_model():
    """Run the original TensorFlow binary-classification example."""
    import tensorflow as tf
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    features, labels = make_classification(n_samples=1000, n_features=20, random_state=42)
    x_train, x_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(20,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return float(accuracy)


def general_ai_model(prompt):
    """Generate up to 100 new GPT-2 tokens; weights load only when called."""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_new_tokens=100, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def self_aware_ai():
    """Return the original theoretical-concept example."""
    return "Self-aware AI is a theoretical concept and not yet achievable with current technology."


def test_reactive_machine():
    assert reactive_machine(3) == "Positive"
    assert reactive_machine(0) == "Negative"
    assert reactive_machine(-2) == "Negative"
