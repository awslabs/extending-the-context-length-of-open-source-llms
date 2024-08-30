import os
import hashlib

html_header = """
<style>
/* Existing Styles for Larger Screens */
.header-container {
  display: flex;
  justify-content: left;
  align-items: center;
  text-align: left;
  background: linear-gradient(45deg, rgba(195, 253, 245, 1), rgba(255, 0, 80, 0.3));
  border-radius: 10px;
  box-shadow: 0 8px 16px 0 rgba(0,0,0,0.1);
  padding: 10px 20px; /* Added padding */
}

.header-container img {
  max-width: 80px;
  height: auto;
  border-radius: 10px;
}

.header-container a {
  color: black; /* Ensure text color is always black */
  text-decoration: none;
}

/* Responsive adjustments for screens less than 768px wide */
@media (max-width: 768px) {
  .header-container {
    flex-direction: column;
    align-items: flex-start;
    padding: 10px 15px; /* Adjust padding for smaller screens */
  }

  .header-container img {
    max-width: 60px; /* Adjust image size for smaller screens */
  }

  .header-container h2, .header-container h5 {
  color: black; /* Ensure text color is always black */
  text-decoration: none;
    text-align: center; /* Center text on small screens */
    margin-top: 5px; /* Add top margin for better spacing after stacking */
  }

  .header-container h2 {
  color: black; /* Ensure text color is always black */
  text-decoration: none;
    font-size: 16px; /* Smaller font size for the title on mobile */
  }

  .header-container h5 {
  color: black; /* Ensure text color is always black */
  text-decoration: none;
    font-size: 12px; /* Smaller font size for the subtitle on mobile */
  }

  .header-container a {
  color: black; /* Ensure text color is always black */
  text-decoration: none;
  }
}
</style>

<div class="header-container">
  <a href="https://w.amazon.com/bin/view/AWS/Teams/SA/Rapid_Prototyping/" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
    <img src="https://d0.awsstatic.com/logos/powered-by-aws.png" alt="LongLLava">
  </a>
  <div>
    <h2><a href="https://w.amazon.com/bin/view/AWS/Teams/SA/Rapid_Prototyping/">Long Context Multimodal LLM Demo by AWS Prototyping Team</a></h2>
  </div>
</div>
"""

block_css = """
#buttons button {
    min-width: min(120px,100%);
}
"""

PARENT_FOLDER = os.path.dirname(os.path.abspath(__file__))
################## BACKEND ##################
os.environ["GRADIO_EXAMPLES_CACHE"] = (
    f"{PARENT_FOLDER}/cache"
)
os.environ["GRADIO_TEMP_DIR"] = (
    f"{PARENT_FOLDER}/cache"
)

def generate_file_hash(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()[:6]