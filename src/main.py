from retrieval_script import get_retrived_data
from generation_script import get_generated_data
import os

topics = ["side_channel_attack", "Information_Leakage"]

output_dir = "generated_outputs"
os.makedirs(output_dir, exist_ok=True)

for topic in topics:
    retrieved_info = get_retrived_data(topic)
    generated_info = get_generated_data(
        retrieved_data= retrieved_info,
        threat= topic,
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1000
    )
    
    output_file = os.path.join(output_dir, f"{topic}_output.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_info)

    print(f"Generated info for '{topic}' saved to {output_file}")
