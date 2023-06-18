import pygame as pg
from types import SimpleNamespace
import numpy as np
import jax
import jax.numpy as jnp
import traceback

wall_color = jnp.asarray([131, 136, 145]) #grey
photosynth_color = jnp.asarray([0, 255, 0]) #green
absorber_color = jnp.asarray([217, 0, 255]) #purple
enzyme_color = jnp.asarray([255, 0, 0]) #red
mover_color = jnp.asarray([0, 0, 255]) #blue
channel_color = jnp.asarray([255, 255, 0]) #yellow

cell_colors = {"wall": wall_color, "photosynth": photosynth_color, "absorber": absorber_color, "enzyme": enzyme_color, "mover": mover_color, "channel": channel_color}

class Visualizer():
    def __init__(self, C, control_frame_time = False, visualize = True):
        self.C = C
        # with visualize disabled, we can still use the visualizer to generate images
        if visualize:
            pg.init()
            self.display = pg.display.set_mode((self.C["screen_size"], self.C["screen_size"]))
            self.font = pg.font.Font('freesansbold.ttf', 24)
        self.selected_view = 0
        self.selected_cell_type = 0
        self.normalize = True
        #Views
        V = SimpleNamespace()
        V.cell_type_view = 0
        V.nutrient_view = 1
        V.organic_matter_view = 2
        V.organic_matter_nutrient_waste_view = 3
        V.individual_cell_type_view = 4
        V.stats_view = 5
        V.spread_view = 6
        V.no_view = 7
        waste_or_energy_string = " and waste" if self.C["enable_waste"] else (" and energy" if self.C["enable_energy"] else "")
        view_names = ["cell type", "nutrient", "organic matter", 
                      "organic matter and nutrient "+waste_or_energy_string, 
                      "individual cell type", "stats", "spread view","no view"]
        num_views = len(view_names)
        if("mover" in self.C["enabled_types"]):
            V.cling_view = num_views
            num_views+=1
            view_names+=["cling"]
        else:
            V.cling_view = -1
        if(self.C["enable_waste"]):
            V.waste_view = num_views
            num_views+=1
            view_names+=["waste"]
        else:
            V.waste_view = -1
        self.V = V
        self.view_names = view_names
        self.num_views = num_views
        self.num_cell_types = len(self.C["enabled_types"])
        self.enabled_types = self.C["enabled_types"]

        #TODO: may be a better way to express this
        cell_type_names = ["wall", "photosynth", "absorber", "enzyme", "mover", "channel"]
        i = 0
        photosynth = absorber = enzyme = mover = channel = -1
        self.cell_type_ids = {}
        for name in cell_type_names:
            if name in self.enabled_types:
                globals()[name] = i
                self.cell_type_ids[name] = i
                i+=1

        self.control_frame_time = control_frame_time
        if control_frame_time:
            self.frame_times = [-2**(-i) for i in range(2,14)][::-1]+[0]+[2**(-i) for i in range(2,14)]
            #get index following zero
            self.frame_time_index = self.frame_times.index(0)+1 

    def handle_user_input(self):
        done = False
        mouse_pos = None
        events = pg.event.get()
        for event in events:
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_LEFT:
                    self.selected_view = (self.selected_view-1)%self.num_views
                if event.key == pg.K_RIGHT:
                    self.selected_view = (self.selected_view+1)%self.num_views
                if event.key == pg.K_UP:
                    self.selected_cell_type = (self.selected_cell_type+1)%self.num_cell_types
                if event.key == pg.K_DOWN:
                    self.selected_cell_type = (self.selected_cell_type-1)%self.num_cell_types
                if event.key == pg.K_q:
                    done = True
                #number keys set self.selected_view
                if event.key in [pg.K_1, pg.K_2, pg.K_3, pg.K_4, pg.K_5, pg.K_6, pg.K_7, pg.K_8, pg.K_9]:
                    self.selected_view = min(event.key - pg.K_1, self.num_views-1)
                #spacebar goes to no view
                if event.key == pg.K_SPACE:
                    self.selected_view = self.V.no_view
                if event.key == pg.K_n:
                    self.normalize = not self.normalize
                if(self.control_frame_time):
                    if event.key == pg.K_d:
                        self.frame_time_index = min((self.frame_time_index+1), len(self.frame_times)-1)
                    elif event.key == pg.K_a:
                        self.frame_time_index = max((self.frame_time_index-1), 0)
                    #s pauses
                    elif event.key == pg.K_s:
                        self.frame_time_index = self.frame_times.index(0)
            elif event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    mouse_pos = event.pos
        if(self.selected_view == self.V.individual_cell_type_view):
            text = self.font.render("selected view: "+self.view_names[self.selected_view]+", selected cell type: "+self.enabled_types[self.selected_cell_type]
                            , True, (255, 255, 255), (0, 0, 0))
        else:
            text = self.font.render("selected view: "+self.view_names[self.selected_view], True, (255, 255, 255), (0, 0, 0))
        self.display.blit(text, (0, 0))
        if(self.control_frame_time):
            frame_time = self.frame_times[self.frame_time_index]
            text = self.font.render("frame_time:{:.5f}".format(frame_time), True, (255, 255, 255), (0, 0, 0))
            self.display.blit(text, (self.C["screen_size"]-250, self.C["screen_size"]-24))
            return done, mouse_pos, frame_time
        else:
            return done, mouse_pos
    
    def nutrient_img(self,S):
        nutrients = S["nutrients"]
        img = np.zeros((self.C["world_size"], self.C["world_size"], 3))
        #nutrient is green
        img[:,:,1] = (255 * jnp.clip(nutrients/self.C["nutrient_cap"], 0, 1))
        return img.astype(np.uint8)

    def waste_img(self, S):
        waste = S["waste"]
        img = np.zeros((self.C["world_size"], self.C["world_size"], 3))
        #waste is blue
        img[:,:,2] = (255 * jnp.clip(waste/self.C["waste_cap"], 0, 1))
        return img.astype(np.uint8)

    def organic_matter_img(self, S):
        organic_matter = S["organic_matter"]
        img = np.zeros((self.C["world_size"], self.C["world_size"], 3))
        #organic matter is red
        img[:,:,0] = (255 * jnp.clip(organic_matter/self.C["organic_matter_hardcap"], 0, 1))
        return img.astype(np.uint8)

    def organic_matter_and_nutrient_img(self, S):
        organic_matter = S["organic_matter"]
        nutrients = S["nutrients"]
        img = np.zeros((self.C["world_size"], self.C["world_size"], 3))
        #organic matter is red
        img[:,:,0] = (255 * jnp.clip(organic_matter/self.C["organic_matter_hardcap"], 0, 1))
        #nutrient is green
        img[:,:,1] = (255 * jnp.clip(nutrients/self.C["nutrient_cap"], 0, 1))
        if(self.C["enable_waste"]):
            #waste is blue
            img[:,:,2] = (255 * jnp.clip(S["waste"]/self.C["waste_cap"], 0, 1))
        elif(self.C["enable_energy"]):
            #energy is blue
            img[:,:,2] = (255 * jnp.clip(S["energy"]/self.C["energy_cap"], 0, 1))
        return img.astype(np.uint8)

    def cell_type_img(self, S):
        cell_type_weights = jax.nn.softmax(S["cell_type_logits"], axis=0)
        if(self.normalize):
            cell_type_weights *= S["organic_matter"]/self.C["organic_matter_hardcap"]
        img = np.zeros((self.C["world_size"], self.C["world_size"], 3))
        for name in self.enabled_types:
            img+=cell_type_weights[self.cell_type_ids[name]][:,:,None]*cell_colors[name]
        return img.astype(np.uint8)

    def individual_cell_type_img(self, S):
        name = self.enabled_types[self.selected_cell_type]
        type_id = self.cell_type_ids[name]
        cell_type_weights = jax.nn.softmax(S["cell_type_logits"], axis=0)
        if(self.normalize):
            cell_type_weights *= S["organic_matter"]/self.C["organic_matter_hardcap"]
        cell_type_weight = cell_type_weights[type_id]
        img = np.zeros((self.C["world_size"], self.C["world_size"], 3))
        #show all organic matter in red, specific cell type in green
        img[:,:,1] = (255 * jnp.clip(cell_type_weight, 0, 1))
        if(self.normalize):
            img[:,:,0] = (255 * jnp.clip((S["organic_matter"]/self.C["organic_matter_hardcap"]-cell_type_weight), 0, 1))
        return img.astype(np.uint8)

    def cling_img(self, S, metrics):
        if(metrics):
            cling_weight = metrics["cling"]
            if(self.normalize):
                cling_weight *= S["organic_matter"]/self.C["organic_matter_hardcap"]
            img = np.zeros((self.C["world_size"], self.C["world_size"], 3))
            img[:,:,1] = (255 * jnp.clip(cling_weight, 0, 1))
            return img.astype(np.uint8)
        else:
            return np.zeros((self.C["world_size"], self.C["world_size"], 3)).astype(np.uint8)
        
    def spread_img(self, S, metrics):
        if(metrics):
            spread_weight = metrics["spread"]
            if(self.normalize):
                spread_weight *= S["organic_matter"]/self.C["organic_matter_hardcap"]
            img = np.zeros((self.C["world_size"], self.C["world_size"], 3))
            img[:,:,1] = (255 * jnp.clip(spread_weight, 0, 1))
            return img.astype(np.uint8)
        else:
            return np.zeros((self.C["world_size"], self.C["world_size"], 3)).astype(np.uint8)

    #visualize network weights with 3 random weight vectors assigned to RGB
    # weights = S["weights"]
    # red_weights = jax.tree_map(lambda x: jax.random.normal(random.PRNGKey(0), x.shape), weights)
    # green_weights = jax.tree_map(lambda x: jax.random.normal(random.PRNGKey(1), x.shape), weights)
    # blue_weights = jax.tree_map(lambda x: jax.random.normal(random.PRNGKey(2), x.shape), weights)
    # # normalize color weights to have magnitude 1
    # red_norm = jnp.sqrt(jax.tree_util.tree_reduce(jnp.add, jax.tree_map(lambda x: jnp.sum(x**2, axis=tuple(range(x.ndim - 2))), red_weights)))
    # green_norm = jnp.sqrt(jax.tree_util.tree_reduce(jnp.add, jax.tree_map(lambda x: jnp.sum(x**2, axis=tuple(range(x.ndim - 2))), green_weights)))
    # blue_norm = jnp.sqrt(jax.tree_util.tree_reduce(jnp.add, jax.tree_map(lambda x: jnp.sum(x**2, axis=tuple(range(x.ndim - 2))), blue_weights)))
    # red_weights = jax.tree_map(lambda x: x/red_norm, red_weights)
    # green_weights = jax.tree_map(lambda x: x/green_norm, green_weights)
    # blue_weights = jax.tree_map(lambda x: x/blue_norm, blue_weights)
    # def weight_img(S):
    #     weights = S["weights"]
    #     #check that weights and red weights have the same tree structure
    #     assert jax.tree_util.tree_structure(weights) == jax.tree_util.tree_structure(red_weights)
    #     weight_norm = jnp.sqrt(jax.tree_util.tree_reduce(jnp.add, jax.tree_map(lambda x: jnp.sum(x**2, axis=tuple(range(x.ndim - 2))), weights)))
    #     red = (jax.tree_util.tree_reduce(jnp.add, jax.tree_util.tree_map(lambda x,y: jnp.sum(x*y, axis=tuple(range(x.ndim - 2)))/weight_norm, weights, red_weights))+1)/2
    #     green = (jax.tree_util.tree_reduce(jnp.add, jax.tree_util.tree_map(lambda x,y: jnp.sum(x*y, axis=tuple(range(x.ndim - 2)))/weight_norm, weights, green_weights))+1)/2
    #     blue = (jax.tree_util.tree_reduce(jnp.add, jax.tree_util.tree_map(lambda x,y: jnp.sum(x*y, axis=tuple(range(x.ndim - 2)))/weight_norm, weights, blue_weights))+1)/2
    #     if(self.normalize):
    #         red = jax.tree_util.tree_map(lambda x: x*S["organic_matter"]/self.C["organic_matter_hardcap"], red)
    #         green = jax.tree_util.tree_map(lambda x: x*S["organic_matter"]/self.C["organic_matter_hardcap"], green)
    #         blue = jax.tree_util.tree_map(lambda x: x*S["organic_matter"]/self.C["organic_matter_hardcap"], blue)
    #     print("red", jnp.sum(red))
    #     print("red_shape", red.shape)
    #     img = np.zeros((self.C["world_size"], self.C["world_size"], 3))
    #     img[:,:,0] = (255 * jnp.clip(red, 0, 1))
    #     img[:,:,1] = (255 * jnp.clip(green, 0, 1))
    #     img[:,:,2] = (255 * jnp.clip(blue, 0, 1))
    #     return img.astype(np.uint8)



    def display_stats(self, S, step, metrics):
        cell_type_weights = S["organic_matter"] * jax.nn.softmax(S["cell_type_logits"], axis=0)
        strings = []
        for name in self.enabled_types:
            type_weight = cell_type_weights[self.cell_type_ids[name]]
            type_frac = jnp.sum(type_weight) / jnp.sum(S["organic_matter"])
            strings.append(name+" frac:{:.3f}".format(type_frac))

        total_nutrients = jnp.sum(S["nutrients"])
        total_organic_matter = jnp.sum(S["organic_matter"])

        if(self.C["enable_waste"]):
            total_waste = jnp.sum(S["waste"])
            strings.append("total_waste:{:.1f}".format(total_waste))
        elif(self.C["enable_energy"]):
            total_energy = jnp.sum(S["energy"])
            strings.append("total_energy:{:.1f}".format(total_energy))

        if(self.C["enable_metaevolution"]):
            avg_mutation_rate = metrics["avg_mutation_rate"]
            strings.append("log_avg_mutation_rate:{:.5f}".format(jnp.log(avg_mutation_rate)))
        strings.append("total_nutrients:{:.1f}".format(total_nutrients))
        strings.append("total_organic_matter:{:.1f}".format(total_organic_matter))
        #weights aren't saved in record so this won't work in replay
        if("weights" in S):
            average_weight_mag = jnp.sqrt(sum(jax.tree_util.tree_map(
                lambda x: jnp.mean(x**2), S["weights"]))/len(S["weights"]))
            strings.append("average_weight_mag:{:.3f}".format(average_weight_mag))
        strings.append("step:{}".format(step))
        for i,s in enumerate(strings):
            text = self.font.render(s, True, (255, 255, 255), (0, 0, 0))
            self.display.blit(text, (0, 24*(i+1)))

    def display_img(self, img):
        surf = pg.surfarray.make_surface(np.asarray(img))
        surf = pg.transform.scale(surf, (self.C["screen_size"], self.C["screen_size"]))
        self.display.blit(surf, (0, 0))

    def update(self, S, step, metrics=None):
        try:
            if(self.selected_view == self.V.cell_type_view):
                img = self.cell_type_img(S)
            elif(self.selected_view == self.V.nutrient_view):
                img = self.nutrient_img(S)
            elif(self.selected_view == self.V.waste_view):
                img = self.waste_img(S)
            elif(self.selected_view == self.V.organic_matter_view):
                img = self.organic_matter_img(S)
            elif(self.selected_view == self.V.organic_matter_nutrient_waste_view):
                img = self.organic_matter_and_nutrient_img(S)
            elif(self.selected_view == self.V.individual_cell_type_view):
                img = self.individual_cell_type_img(S)
            elif(self.selected_view == self.V.cling_view):
                img = self.cling_img(S, metrics)
            elif(self.selected_view == self.V.spread_view):
                img = self.spread_img(S, metrics)
            # elif(self.selected_view == weight_view):
            #     img = weight_img(S, normalize)
            elif(self.selected_view == self.V.stats_view):
                #clear display
                self.display.fill((0,0,0))
                self.display_stats(S, step, metrics)
            elif((self.selected_view == self.V.no_view)):
                #clear display
                self.display.fill((0,0,0))
            if((self.selected_view not in [self.V.stats_view, self.V.no_view])):
                self.display_img(img)
        except Exception as e:
            exception_str = traceback.format_exc()
            #split into seperate strings
            exception_strs = ["Exception in generating view!"]+[exception_str[i:i+50] for i in range(0, len(exception_str), 50)]
            text = self.font.render(exception_str, True, (255, 255, 255), (0, 0, 0))
            for i, s in enumerate(exception_strs):
                text = self.font.render(s, True, (255, 255, 255), (0, 0, 0))
                self.display.blit(text, (0, 24*(i+1)))
        if(self.control_frame_time):
            done, mouse_pos, frame_time = self.handle_user_input()
        else:
            done, mouse_pos = self.handle_user_input()
        if(mouse_pos):
            try:
                print("\n\nmouse_pos: ", mouse_pos)
                #draw a green circle at mouse position
                pg.draw.circle(self.display, (0,255,0), mouse_pos, 5)
                #display cell type at mouse position
                cell_type_weights = jax.nn.softmax(S["cell_type_logits"], axis=0)
                #convert mouse position to cell position
                cell_pos = (int(mouse_pos[0]/self.C["screen_size"]*self.C["world_size"]), int(mouse_pos[1]/self.C["screen_size"]*self.C["world_size"]))
                print("cell_pos: ", cell_pos)
                total_matter = S["organic_matter"][cell_pos[0], cell_pos[1]]
                print("total matter: ", total_matter)
                for name in self.enabled_types:
                    print(name, cell_type_weights[self.cell_type_ids[name]][cell_pos[0], cell_pos[1]], sep=": ")
                print("nutrients: ", S["nutrients"][cell_pos[0], cell_pos[1]])
                if(self.C["enable_waste"]):
                    print("waste: ", S["waste"][cell_pos[0], cell_pos[1]])
                elif(self.C["enable_energy"]):
                    print("energy: ", S["energy"][cell_pos[0], cell_pos[1]])
                if("mover" in self.enabled_types):
                    print("cling: ", metrics["cling"][cell_pos[0], cell_pos[1]])
                print("spread_weight: ", metrics["spread"][cell_pos[0], cell_pos[1]])
            except Exception as e:
                print("Exception in displaying cell info!")
                print(traceback.format_exc())
            # for i in range(num_cell_types):
            #     print(cell_type_names[i], cell_type_weights[i, cell_pos[0], cell_pos[1]], sep=": ")
        pg.display.update()
        if(self.control_frame_time):
            return done, frame_time
        else:
            return done