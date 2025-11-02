"""Rep counter logic for tracking exercise repetitions."""


class RepCounter:
    """Tracks repetitions based on angle thresholds."""
    
    def __init__(self, up_threshold=160, down_threshold=70):
        """
        Initialize the rep counter.
        
        Args:
            up_threshold: Angle threshold for "up" position (degrees)
            down_threshold: Angle threshold for "down" position (degrees)
        """
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        self.rep_count = 0
        self.state = "down"
    
    def update(self, angle):
        """
        Update the counter based on current angle.
        
        Args:
            angle: Current joint angle in degrees
        
        Returns:
            bool: True if a new rep was counted, False otherwise
        """
        new_rep = False
        
        if angle > self.up_threshold:
            self.state = "up"
        elif angle < self.down_threshold and self.state == "up":
            self.state = "down"
            self.rep_count += 1
            new_rep = True
        
        return new_rep
    
    def reset(self):
        """Reset the counter to initial state."""
        self.rep_count = 0
        self.state = "down"
    
    def get_count(self):
        """Get the current rep count."""
        return self.rep_count
    
    def get_state(self):
        """Get the current state."""
        return self.state
