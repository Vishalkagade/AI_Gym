"""Analyze bicep curl data from CSV file."""
import pandas as pd
import matplotlib.pyplot as plt


def analyze_workout_data(csv_file='bicep_curl_data.csv'):
    """
    Analyze and visualize workout data from CSV.
    
    Args:
        csv_file: Path to the CSV file
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Filter only frames where pose was detected
    df_detected = df[df['pose_detected'] == True].copy()
    
    # Convert angle to numeric (handle 'N/A' values)
    df_detected['angle'] = pd.to_numeric(df_detected['angle'], errors='coerce')
    
    print("=" * 60)
    print("WORKOUT SUMMARY")
    print("=" * 60)
    print(f"Total frames captured: {len(df)}")
    print(f"Frames with pose detected: {len(df_detected)} ({len(df_detected)/len(df)*100:.1f}%)")
    print(f"Total reps completed: {df['rep_count'].max()}")
    print(f"Average angle: {df_detected['angle'].mean():.2f}°")
    print(f"Min angle (max contraction): {df_detected['angle'].min():.2f}°")
    print(f"Max angle (max extension): {df_detected['angle'].max():.2f}°")
    
    # Count time in each state
    state_counts = df['state'].value_counts()
    print(f"\nTime distribution:")
    for state, count in state_counts.items():
        print(f"  {state}: {count} frames ({count/len(df)*100:.1f}%)")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Bicep Curl Analysis', fontsize=16)
    
    # Plot 1: Angle over time
    axes[0, 0].plot(df_detected['frame_number'], df_detected['angle'], 'b-', alpha=0.7)
    axes[0, 0].axhline(y=160, color='r', linestyle='--', label='Up threshold (160°)')
    axes[0, 0].axhline(y=70, color='g', linestyle='--', label='Down threshold (70°)')
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('Angle (degrees)')
    axes[0, 0].set_title('Arm Angle Over Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Rep count over time
    axes[0, 1].plot(df['frame_number'], df['rep_count'], 'g-', linewidth=2)
    axes[0, 1].set_xlabel('Frame Number')
    axes[0, 1].set_ylabel('Rep Count')
    axes[0, 1].set_title('Cumulative Reps')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Angle distribution
    axes[1, 0].hist(df_detected['angle'], bins=30, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(x=160, color='r', linestyle='--', label='Up threshold')
    axes[1, 0].axvline(x=70, color='g', linestyle='--', label='Down threshold')
    axes[1, 0].set_xlabel('Angle (degrees)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Angle Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: State timeline
    state_map = {'up': 1, 'down': 0}
    df['state_numeric'] = df['state'].map(state_map)
    axes[1, 1].fill_between(df['frame_number'], 0, df['state_numeric'], 
                             alpha=0.5, color='orange', label='State')
    axes[1, 1].set_xlabel('Frame Number')
    axes[1, 1].set_ylabel('State')
    axes[1, 1].set_yticks([0, 1])
    axes[1, 1].set_yticklabels(['Down', 'Up'])
    axes[1, 1].set_title('State Timeline')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('workout_analysis.png', dpi=300, bbox_inches='tight')
    print("\n" + "=" * 60)
    print("Visualization saved as 'workout_analysis.png'")
    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    import sys
    
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'bicep_curl_data.csv'
    
    try:
        analyze_workout_data(csv_file)
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")
        print("Please run the main application first to generate workout data.")
    except Exception as e:
        print(f"Error analyzing data: {e}")
