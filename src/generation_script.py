from api_handler import call_openai_api

def get_generated_data(retrieved_data, threat, model, temperature, max_tokens):

    prompt = f"""

    You are provided with a set of retrieved information below.

    =========================
    Retrieved Data:
    {retrieved_data}
    =========================

    The above data contains information about the threat: **{threat}**.

    Your task is to analyze the data and generate a comprehensive, structured, and informative explanation of the different **attack scenarios** associated with this threat. Your response will be used as background context for threat modeling in hardware security verification.

    Please include the following in your response:

    1. **Threat Description**: Briefly explain what the threat "{threat}" means in the context of hardware systems.

    2. **Attack Scenarios**:
    - List different plausible **attack scenarios** related to this threat.
    - For each scenario, describe:
        - **How the attack is conducted** (mechanism or process).
        - **Target hardware IPs or components** (e.g., processor core, memory, interconnect, etc.).
        - **Potential impact** on system behavior or security.
        - **Real-world examples or theoretical models**, if applicable.

    3. **Attacker Profile**:
    - What kind of access, capabilities, or tools would an attacker need for each scenario?

    4. **Detection or Indicators**:
    - Mention any typical signs or anomalies that may help in detecting such an attack (optional, if present in data).

    Ensure your response is **well-organized and technical**, with a focus on how each scenario can be directly applicable to attacking specific hardware IPs. The goal is to provide the LLM with rich, structured, and precise background knowledge about this threat.

    """


    return call_openai_api(prompt, model, temperature, max_tokens)