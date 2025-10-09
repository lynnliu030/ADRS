#!/usr/bin/env python3
"""
Helper script to inject research paper content into OpenEvolve config
"""

import yaml
from pathlib import Path

def load_research_context(research_file: str = "research_context.md") -> str:
    """Load research context from external file"""
    try:
        with open(research_file, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Warning: {research_file} not found. Using placeholder.")
        return "[Please add your research paper content to research_context.md]"

def update_config_with_research(config_file: str = "config.yaml", research_file: str = "research_context.md"):
    """Update config.yaml with research context from external file"""
    
    # Load research content
    research_content = load_research_context(research_file)
    
    # Load config
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get current system message
    current_system_msg = config['prompt']['system_message']
    
    # Replace placeholder with actual content
    updated_system_msg = current_system_msg.replace(
        "[PASTE YOUR EXTRACTED PDF TEXT HERE]", 
        research_content
    )
    
    # Update config
    config['prompt']['system_message'] = updated_system_msg
    
    # Write back to config
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, width=120, allow_unicode=True)
    
    print(f"✅ Updated {config_file} with content from {research_file}")
    print(f"📄 Research content length: {len(research_content)} characters")

if __name__ == "__main__":
    update_config_with_research() 