class PassengerCounter:

    def __init__(self, capacity=30):

        self.capacity = capacity

        self.passengers = 0
        self.entered = 0
        self.exited = 0

        # Store the last known position of each person
        self.previous_x = {}

        # Store whether the person has entered the counting zone
        self.in_counting_zone = {}

    def update(self, person_id, center_x, line_x):

        event = None

        # =========================================
        # First time seeing this person
        # =========================================

        if person_id not in self.previous_x:

            self.previous_x[person_id] = center_x
            self.in_counting_zone[person_id] = False

            return None

        previous_x = self.previous_x[person_id]

        # =========================================
        # Define zones
        # =========================================

        left_boundary = line_x - 50
        right_boundary = line_x + 50

        # =========================================
        # Person enters counting zone
        # =========================================

        if left_boundary <= center_x <= right_boundary:

            self.in_counting_zone[person_id] = True

        # =========================================
        # ENTER: LEFT -> RIGHT
        # =========================================

        if (
            self.in_counting_zone.get(person_id, False)
            and previous_x <= right_boundary
            and center_x > right_boundary
        ):

            # Make sure this movement started from the left
            if previous_x < line_x:

                if self.passengers < self.capacity:

                    self.passengers += 1
                    self.entered += 1

                    event = "IN"

                else:

                    event = "FULL"

                self.in_counting_zone[person_id] = False

        # =========================================
        # EXIT: RIGHT -> LEFT
        # =========================================

        elif (
            self.in_counting_zone.get(person_id, False)
            and previous_x >= left_boundary
            and center_x < left_boundary
        ):

            # Make sure this movement started from the right
            if previous_x > line_x:

                self.passengers -= 1

                self.passengers = max(
                    self.passengers,
                    0
                )

                self.exited += 1

                event = "OUT"

                self.in_counting_zone[person_id] = False

        # =========================================
        # Update position
        # =========================================

        self.previous_x[person_id] = center_x

        return event

    def get_available(self):

        return max(
            self.capacity - self.passengers,
            0
        )

    def get_status(self):

        if self.passengers >= self.capacity:

            return "BUS FULL"

        elif self.passengers >= self.capacity * 0.9:

            return "ALMOST FULL"

        else:

            return "NOT FULL"