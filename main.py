import argparse
from MAC_KG import MAC_KG

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MAC-KG')

    # ============ General Settings ==============
    parser.add_argument('--start_entity', default='giant panda', help='The starting entity for the multi-modal knowledge graph')
    parser.add_argument('--textual_base', default='./textual_base', help='Warehouse for textual data')
    parser.add_argument('--visual_base', default='./visual_base', help='Warehouse for visual images')
    parser.add_argument('--hop', default=1, help='The number of iterations to generate triples')
    parser.add_argument('--max_neighbors', default=3, help='At most max_neighbors neighbor nodes are selected for the next iteration.')
    parser.add_argument('--store_dir', default='./MAC_KG_store', help='The directory to store the constructed multi-modal knowledge graph')

    # ============ Content Crawler Settings ==============
    parser.add_argument('--content_crawler_user_agent', default='MyWikiProject/1.0 (your_email@email.com)',
                        help='Please replace it with your own User-Agent information')


    # ============ Agent API Settings ==============

    # Corpus Extracting Agent
    parser.add_argument('--CEA_api_key', default='API KEY',
                        help='API key')
    parser.add_argument('--CEA_url', default='API URL', help='Optional API base URL')
    parser.add_argument('--CEA_model', default='gpt-4o', help='model name')


    # Textual Enhancing Agent
    parser.add_argument('--TEA_api_key', default='API KEY',
                        help='API key')
    parser.add_argument('--TEA_url', default='API URL', help='Optional API base URL')
    parser.add_argument('--TEA_model', default='gpt-4o', help='model name')


    # Structure Verifying Agent
    parser.add_argument('--SVA_api_key', default='API KEY',
                        help='API key')
    parser.add_argument('--SVA_url', default='API URL', help='Optional API base URL')
    parser.add_argument('--SVA_model', default='gpt-4o', help='model name')


    # Triple Aligning Agent
    parser.add_argument('--TAA_model', default='./models/sentence-transformers/all-MiniLM-L6-v2',
                        help='Triple Aligning Model')
    parser.add_argument('--TAA_threshold', default=0.8, type=float, help='Triple Aligning Threshold')


    # Tail Pruning Agent
    parser.add_argument('--TPA_api_key', default='API KEY',
                        help='API key')
    parser.add_argument('--TPA_url', default='API URL', help='Optional API base URL')
    parser.add_argument('--TPA_model', default='gpt-4o', help='Tail pruning model name')


    # Image Filtering Agent
    parser.add_argument('--IFA_model', default='./models/CLIP/clip-vit-base-patch32',
                        help='The model used for image filtering')


    # Visual Enhancing Agent
    # Visual Enhancing Agent Image to Description Module
    parser.add_argument('--VEA_I2D_api_key', default='API KEY',
                        help='API key')
    parser.add_argument('--VEA_I2D_url', default='API URL', help='Optional API base URL')
    parser.add_argument('--VEA_I2D_model', default='gpt-4o', help='model name')


    # Visual Enhancing Agent Description to Triple Module
    parser.add_argument('--VEA_D2T_api_key', default='API KEY',
                        help='API key')
    parser.add_argument('--VEA_D2T_url', default='API URL', help='Optional API base URL')
    parser.add_argument('--VEA_D2T_model', default='gpt-4o', help='model name')

    # Image2text Matching Agent
    parser.add_argument('--IMA_model', default='./models/CLIP/clip-vit-base-patch32',
                        help='The model used for image-text matching')
    parser.add_argument('--IMA_topK', default=20, type=int, help='Top K image2text similar images to be selected for each entity')

    # Image Clustering Agent
    parser.add_argument('--ICA_model', default='./models/CLIP/clip-vit-base-patch32',
                        help='The model used for image clustering')
    parser.add_argument('--ICA_clusters', default=2, type=int, help='Number of clusters for image clustering')


    # ============ Running ==============
    args = parser.parse_args()
    print(args)

    # Initialize MAC
    mac_kg = MAC_KG(args)

    # Start automatic construction
    mac_kg.mmkg_automatic_construction()

    # Summarize the constructed MMKG
    mac_kg.summarize_constructed_mmkg()
