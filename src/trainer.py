import numpy as np
import torch
import datetime
import time

class Trainer:

    def __init__(self, model, optimizer, scheduler, train_data, valid_data, args):
        super(Trainer, self).__init__()
        self.start = time.time()
        self.optimizer = optimizer
        self.train_data = train_data
        self.valid_data = valid_data
        self.args = args
        self.device = args.device
        self.model = model
        self.epochs = args.epochs
        self.scheduler = scheduler
        self.best_acc = -1

    
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def run(self):
        self.model.zero_grad()
        for epoch_i in range(0, self.epochs):
            
            # ========================================
            #               Training
            # ========================================
            
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, self.epochs))
            print('Training...')

            t0 = time.time()
            total_loss = 0
            self.model.train()

            for step, batch in enumerate(self.train_data):
                if step % self.args.checkpoint == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(self.train_data), elapsed))
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                outputs = self.model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)

                loss = outputs[0]
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.model.zero_grad()

            avg_train_loss = total_loss / len(self.train_data)            

            print("")
            print("  Average training loss: {0:.3f}".format(avg_train_loss))
            print("  Training epoch took: {:}".format(self.format_time(time.time() - t0)))
                
            # ========================================
            #               Validation
            # ========================================

            print("")
            print("Running Validation...")

            t0 = time.time()
            self.model.eval()
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            for batch in self.valid_data:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():     
                    outputs = self.model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask)
                
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()
                
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            acc = eval_accuracy / nb_eval_steps
            print("  Accuracy: {0:.3f}".format(eval_accuracy/nb_eval_steps))
            print("  Validation took: {:}".format(self.format_time(time.time() - t0)))
            if self.best_acc < acc:
                print("Best model saved")
                self.best_acc = acc
                torch.save(self.model, self.args.save_dir)

        print("")
        print("Training complete!")
