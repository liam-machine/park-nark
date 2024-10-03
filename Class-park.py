from datetime import datetime, timedelta
import time
# Global dictionary to track which car is parked in which car park
global_parking_registry = {}

class Car:
    def __init__(self, car_id):
        self.car_id = car_id

    def __str__(self):
        return f"Car ID: {self.car_id}"

class CarPark:
    # Class-level dictionary to track all car park instances
    car_park_registry = {}

    def __init__(self, park_id):
        self.park_id = park_id
        self.parking_records = {}  # Dictionary to store parking records
        self.current_car = None  # Track the current car in the car park
        CarPark.car_park_registry[park_id] = self

    def park_car(self, car):
        # Check if the car is already parked in another car park
        if car.car_id in global_parking_registry:
            print(f"Car {car.car_id} is already parked in {global_parking_registry[car.car_id]}.")
            return

        # Check if the car park is already occupied
        if self.current_car is not None:
            print(f"Car park {self.park_id} is already occupied by car {self.current_car}.")
            return

        entry_time = datetime.now()
        self.parking_records[car.car_id] = {
            'entry_time': entry_time,
            'exit_time': None
        }
        global_parking_registry[car.car_id] = self.park_id
        self.current_car = car.car_id
        print(f"Car {car.car_id} parked at {entry_time.strftime('%Y-%m-%d %H:%M:%S')} in {self.park_id}")

    def leave_car(self, car):
        if car.car_id in self.parking_records and self.parking_records[car.car_id]['exit_time'] is None:
            exit_time = datetime.now()
            self.parking_records[car.car_id]['exit_time'] = exit_time
            del global_parking_registry[car.car_id]
            self.current_car = None
            print(f"Car {car.car_id} left at {exit_time.strftime('%Y-%m-%d %H:%M:%S')} from {self.park_id}")
        else:
            print(f"Car {car.car_id} is not currently parked in {self.park_id}.")

    def get_parking_record(self, car):
        if car.car_id in self.parking_records:
            record = self.parking_records[car.car_id]
            entry_time = record['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
            exit_time = record['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if record['exit_time'] else 'Still parked'
            return f"Car {car.car_id} parked at {entry_time}, left at {exit_time}"
        else:
            return f"{self.park_id}: No record found for Car {car.car_id}"

    def print_history(self):
        print(f"History of Car Park {self.park_id}:")
        for car_id, record in self.parking_records.items():
            entry_time = record['entry_time'].strftime('%Y-%m-%d %H:%M:%S')
            exit_time = record['exit_time'].strftime('%Y-%m-%d %H:%M:%S') if record['exit_time'] else 'Still parked'
            print(f"Car {car_id} parked at {entry_time}, left at {exit_time}")

    @classmethod
    def get_all_car_park_status(cls):
        status = []
        for park_id, car_park in cls.car_park_registry.items():
            if car_park.current_car:
                entry_time = car_park.parking_records[car_park.current_car]['entry_time']
                parked_duration = datetime.now() - entry_time
                status.append(f"Car Park {park_id} is occupied by Car {car_park.current_car} for {parked_duration}.")
            else:
                status.append(f"Car Park {park_id} is empty.")
        return status

# Example usage
if __name__ == "__main__":
    car_park1 = CarPark("Park1")
    car_park2 = CarPark("Park2")
    car_park3 = CarPark("Park3")
    car_park4 = CarPark("Park4")
    car_park5 = CarPark("Park5")
    car_park6 = CarPark("Park6")
    car_park7 = CarPark("Park7")
    car_park8 = CarPark("Park8")
    car_park9 = CarPark("Park9")
    car_park10 = CarPark("Park10")
    car_park11 = CarPark("Park11")
    car_park12 = CarPark("Park12")
    car1 = Car("ABC123")
    car2 = Car("XYZ789")
    car3 = Car("LMN456")
    car4 = Car("DEF321")
    car5 = Car("GHI654")
    car6 = Car("JKL987")
    car7 = Car("MNO654")

    car_park1.park_car(car3)
    car_park1.park_car(car4)  # This should print a message that the car park is already occupied
    car_park2.park_car(car4)  # This should successfully park car4 in car_park2
    time.sleep(20)
    car_park2.park_car(car5)  # This should print a message that the car park is already occupied
    car_park3.park_car(car1)
    car_park4.park_car(car2)
    car_park5.park_car(car6)
    time.sleep(20)
    car_park6.park_car(car7)
    car_park7.park_car(car1)  # This should print a message that car1 is already parked in car_park3
    car_park8.park_car(car2)  # This should print a message that car2 is already parked in car_park4
    car_park9.park_car(car3)  # This should print a message that car3 is already parked in car_park1
    time.sleep(20)
    car_park10.park_car(car4)  # This should print a message that car4 is already parked in car_park2
    car_park11.park_car(car5)
    car_park12.park_car(car6)  # This should print a message that car6 is already parked in car_park5

    # Simulate some time passing
    time.sleep(2)

    car_park3.leave_car(car1)
    car_park4.leave_car(car2)
  
    # Simulate some time passing
    import time
    time.sleep(15)

    car_park1.leave_car(car3)

    # Print the entire history of car park 1
    car_park1.print_history()
    print("statuses")
    # Get the status of all car park instances
    all_status = CarPark.get_all_car_park_status()
    for status in all_status:
        print(status)